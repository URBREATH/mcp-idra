import os
import json
from ollama import Client
from orion_server import get_orion_entities

from config import ollama_client, PILOT_CITIES, IDRA_TAG

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


def chat_loop(model_name):
    print("=== 🏙️ IDRA Data Analyst CLI (Powered by MCP & Ollama) ===")
    print(f"Active Model: {model_name}")
    print(f"Active Pilot Cities: {PILOT_CITIES}")
    print("Type 'exit' or 'quit' to close the session.\n")

    # STRICT SYSTEM PROMPT IN ENGLISH
    system_prompt = f"""You are the Data Analyst for the IDRA Smart City platform. 

    STRICT RULES:
    1. MANDATORY RESEARCH: If the user mentions a pilot city from this list: [{PILOT_CITIES}], or [{IDRA_TAG}] even in a generic question, you MUST ALWAYS use the get_orion_entities tool first (passing entity_type="*" and the city name) to check available data. NOTE: The user might type the city in uppercase or lowercase.
    2. DATA-DRIVEN RESPONSES: Integrate the real data retrieved from the database into your answers. If you find traffic data, suggest mobility actions. If the database returns no results, state it clearly.
    3. NO HALLUCINATIONS: Do not generate HTML code, fake web links, or fictional data. Do not provide generic historical/tourist info unless explicitly requested.
    """

    messages = [{'role': 'system', 'content': system_prompt}]

    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            messages.append({'role': 'user', 'content': user_input})

            # 1. Request to Ollama (Temperature 0.0 for deterministic tool usage)
            response = ollama_client.chat(
                model=model_name,
                messages=messages,
                tools=[orion_tool],
                options={'temperature': 0.0, 'num_thread': 12, 'num_ctx': 16384}
            )

            message = response.get('message', {})
            content = message.get('content', '').strip()
            tool_calls = message.get('tool_calls', [])

            # JSON Autocorrection block if the model prints JSON instead of invoking the tool
            if not tool_calls and content.startswith('[') and 'get_orion_entities' in content:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list) and 'name' in parsed[0]:
                        tool_calls = [{'function': parsed[0]}]
                        message['tool_calls'] = tool_calls
                        message['content'] = ""
                except Exception:
                    pass

            # 2. Tool Execution
            if tool_calls:
                for tool in tool_calls:
                    if tool['function']['name'] == 'get_orion_entities':
                        args = tool['function'].get('arguments', {})
                        if isinstance(args, str):
                            args = json.loads(args)

                        e_type = args.get('entity_type', '*')
                        city = args.get('city', None)
                        limit = args.get('limit', 5)

                        target_print = f" for '{city}'" if city else ""
                        print(f"\n[⚙️ Executing MCP Tool: Extracting '{e_type}'{target_print} (max {limit})...]")

                        db_result = get_orion_entities(e_type, city, limit)

                        messages.append(message)
                        messages.append({'role': 'tool', 'content': db_result, 'name': 'get_orion_entities'})

                        final_response = ollama_client.chat(
                            model=model_name,
                            messages=messages,
                            options={'temperature': 0.0}
                        )
                        print(f"\n🤖 Agent: {final_response['message']['content']}\n")
                        messages.append(final_response['message'])
            else:
                # 3. Direct response (No tool required)
                print(f"\n🤖 Agent: {content}\n")
                messages.append(message)

        except Exception as e:
            print(f"\n[❌ Error: {str(e)}]\n")
