
from models import choose_model, prepare_model
from cli_agent import chat_loop


def main():
    ref, source = choose_model()
    prepare_model(ref, source)
    chat_loop(ref)


if __name__ == "__main__":
    main()