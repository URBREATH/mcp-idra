import json

from sympy.physics.units import temperature

from orion_server import get_orion_entities
from config import ollama_client, PILOT_CITIES, IDRA_TAG
import re

orion_tool = {
    'type': 'function',
    'function': {
        'name': 'get_orion_entities',
        'description': 'Extracts real data (datasets, distributions, entities) from the ORION MongoDB database.',
        'parameters': {
            'type': 'object',
            'properties': {
                'entity_type': {
                    'type': 'string',
                    'description': 'NGSI-LD entity type. Use "*" to explore all data of a city. Use "DistributionDCAT-AP" for distributions and "Dataset" for datasets.'
                },
                'city': {
                    'type': 'string',
                    'description': f'City to filter. Valid pilot cities are: {PILOT_CITIES}.'
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Maximum number of results (default 5)'
                }
            },
            'required': ['entity_type']
        }
    }
}

SYSTEM_PROMPT = f"""You are the Data Analyst for the IDRA Smart City platform.

STRICT RULES:
1. MANDATORY RESEARCH: If the user mentions a pilot city from this list: [{PILOT_CITIES}], or [{IDRA_TAG}] even in a generic question, you MUST ALWAYS use the get_orion_entities tool first (passing entity_type="*" and the city name) to check available data. NOTE: the user might type the city in uppercase or lowercase.
2. DATA-DRIVEN RESPONSES: Integrate the real data retrieved from the database into your answers. If you find traffic data, suggest mobility actions. If the database returns no results, state it clearly.
3. NO HALLUCINATIONS: Do not generate HTML code, fake web links, or fictional data. Do not provide generic historical/tourist info unless explicitly requested.
"""


class OllamaBackend:
    def __init__(self, model):
        self.model = model
        self.client = ollama_client  # da config: rispetta OLLAMA_HOST

    def chat(self, messages, tools=None):
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"temperature": 0.0, "num_thread": 12, "num_ctx": 16384},
        )
        msg = resp["message"]
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            fn = tool_calls[0]["function"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            return {"assistant_msg": msg,
                    "tool_call": {"name": fn["name"], "arguments": args},
                    "content": ""}
        return {"assistant_msg": msg,
                "tool_call": None,
                "content": (msg.get("content") or "").strip()}

    def tool_message(self, name, content):
        return {"role": "tool", "tool_name": name, "content": content}


class TransformersBackend:
    def __init__(self, model_path):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype="auto"
        )

    def chat(self, messages, tools=None):
        inputs = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

        call = self._parse_tool_call(text)
        assistant_msg = {"role": "assistant", "content": text}
        if call:
            return {"assistant_msg": assistant_msg, "tool_call": call, "content": ""}
        return {"assistant_msg": assistant_msg, "tool_call": None, "content": text}

    def tool_message(self, name, content):
        return {"role": "tool", "name": name, "content": content}

    @staticmethod
    def _parse_tool_call(text):
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if m:
            payload = m.group(1)
        else:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            payload = m.group(0)

        try:
            data = json.loads(payload)
        except Exception:
            return None

        if isinstance(data, dict) and data.get("name") == "get_orion_entities":
            args = data.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            return {"name": data["name"], "arguments": args}
        return None


def make_backend(serving_ref, backend):
    if backend == "ollama":
        return OllamaBackend(serving_ref)
    return TransformersBackend(serving_ref)

def chat_loop(backend, model_label):
    print("=== 🏙️ IDRA Data Analyst CLI (Powered by MCP & Ollama) ===")
    print(f"Active Model: {model_label}")
    print(f"Backend: {type(backend).__name__}")
    print(f"Active Pilot Cities: {PILOT_CITIES}")
    print("Type 'exit' or 'quit' to close the session.\n")


    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            messages.append({'role': 'user', 'content': user_input})

            result = backend.chat(messages, tools=[orion_tool])

            if result["tool_call"]:
                messages.append(result["assistant_msg"])
                name = result["tool_call"]["name"]
                args = result["tool_call"]["arguments"]
                print(f"\n[Tool: {name} -> {args}]")

                db_result = get_orion_entities(
                    args.get("entity_type", "*"),
                    args.get("city"),
                    args.get("limit", 5),
                )
                messages.append(backend.tool_message(name, db_result))

                final = backend.chat(messages, tools=[orion_tool])
                print(f"\nAgent: {final['content']}\n")
                messages.append(final["assistant_msg"])
            else:
                print(f"\nAgent: {result['content']}\n")
                messages.append(result["assistant_msg"])

        except Exception as e:
            print(f"\n[Errore: {e}]\n")
