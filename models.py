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


def has_model(name):
    """True se il modello e' gia' registrato in Ollama."""
    names = [m.model for m in ollama_client.list().models]
    return name in names


def import_into_ollama(name, repo):
    """Importa un modello HF ad architettura STANDARD dentro Ollama
    (conversione Safetensors->GGUF lato server).

    ATTENZIONE: non adatto alle architetture personalizzate (es. EngGPT2),
    che la conversione interna di Ollama non sa gestire. Quei modelli vanno
    serviti via transformers, non importati qui.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo, local_dir=f"./models/{name}", token=os.getenv("HF_TOKEN")
    )

    files = {}
    for entry in os.scandir(local_dir):
        if entry.is_file() and not entry.name.startswith("."):
            print(f"  upload {entry.name}...")
            files[entry.name] = ollama_client.create_blob(entry.path)

    print(f"Creating model '{name}' in Ollama...")
    for progress in ollama_client.create(model=name, files=files, stream=True):
        print(f"  {progress.status}")
    print("Done.")


def download_for_transformers(repo):
    """Scarica i pesi in locale per il caricamento via transformers.
    NON importa nulla in Ollama. snapshot_download e' idempotente:
    se i file ci sono gia', verifica soltanto.
    Restituisce il percorso locale da passare a from_pretrained.
    """
    from huggingface_hub import snapshot_download

    local_dir = f"./models/{repo.replace('/', '__')}"
    snapshot_download(repo, local_dir=local_dir, token=os.getenv("HF_TOKEN"))
    return local_dir


def prepare_model(ref, source, backend):
    """Prepara il modello coerentemente col backend con cui verra' servito.

    Restituisce il 'riferimento di servizio' da passare a make_backend:
      - per Ollama: il nome del modello in Ollama
      - per transformers: il percorso locale dei pesi
    """
    if backend == "ollama":
        if has_model(ref):
            print(f"OK: '{ref}' is already on Ollama.")
            return ref
        if source == "registry":
            print(f"Pull '{ref}' from Ollama registry...")
            ollama_client.pull(ref)
        else:
            print(f"Import '{ref}' from HuggingFace ({source}) into Ollama...")
            import_into_ollama(ref, source)
        return ref

    print(f"Preparing weights of '{source}' for transformers...")
    return download_for_transformers(source)


def choose_model():
    print("Choose a model:")
    for k, (label, ref, source, backend) in MODELS.items():
        print(f"  [{k}] {label}  ({backend})")

    while True:
        c = input("Chosen model: ").strip()

        if c in MODELS:
            label, ref, source, backend = MODELS[c]
            print(f"-> {label}")
            return ref, source, backend

        if c.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)

        print("Unvalid choice. Retry.")