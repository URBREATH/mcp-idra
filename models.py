# -*- coding: utf-8 -*-
"""Scelta del modello all'avvio e preparazione (download / registrazione)."""

import os
import sys

from dotenv import load_dotenv
from config import ollama_client, MODELS

load_dotenv()

def _kind_of(name):

    for _, (_label, ref, kind) in MODELS.items():
        if ref == name:
            return kind
    return "registry"


def import_from_hf(name, repo):

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo, local_dir=f"./models/{name}",
                                  token=os.getenv("HF_TOKEN"))

    # 1. Carica ogni file del modello come blob sul server -> {nome_file: digest}
    files = {}
    for entry in os.scandir(local_dir):
        if entry.is_file() and not entry.name.startswith("."):   # salta .cache/.gitattributes
            print(f"  upload {entry.name}...")
            files[entry.name] = ollama_client.create_blob(entry.path)

    # 2. Crea il modello: la conversione Safetensors->GGUF avviene lato server
    print(f"Creating model '{name}'...")
    for progress in ollama_client.create(model=name, files=files, stream=True):
        print(f"  {progress.status}")
    print("Done.")

def model_present(name):
    """True se il modello e' gia' scaricato in locale."""
    names = [m.model for m in ollama_client.list().models]
    return name in names or f"{name}" in names


def prepare_model(name, source):
    """Scarica il modello se manca: da registry Ollama o da HuggingFace."""
    if model_present(name):
        print(f"OK: '{name}' is already on the machine.")
        return
    if source == "registry":
        print(f"Download '{name}' from Ollama registry...")
        ollama_client.pull(name)
    else:
        print(f"Download '{name}' from HuggingFace ({source})...")
        import_from_hf(name, source)
    print("Pronto.")


def choose_model():
    print("Choose one model:")
    for k, (label, ref, source) in MODELS.items():
        print(f"  [{k}] {label}")

    while True:
        c = input("Chosen model: ").strip()

        if c in MODELS:
            label, ref, source = MODELS[c]
            print(f"-> {label}")
            return ref, source

        if c.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            sys.exit(0)

        print("Invalid choice. Riprova.")