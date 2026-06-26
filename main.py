from dotenv import load_dotenv

from models import choose_model, prepare_model
from cli_agent import chat_loop, make_backend

load_dotenv()

def main():
    ref, source, backend = choose_model()

    serving_ref = prepare_model(ref, source, backend)
    backend_obj = make_backend(serving_ref, backend)
    chat_loop(backend_obj, serving_ref)

if __name__ == "__main__":
    main()