# -*- coding: utf-8 -*-
"""Configurazione centralizzata e client Ollama condiviso."""

import os
from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PILOT_CITIES = "Aarhus,Athens,Cluj-Napoca,Kajaani,Leuven,Madrid,Parma,Pilsen,Tallinn,Attica,Napoli,Vilnius,Molina"
IDRA_TAG = "IDRA"

MODELS = {
    "1": ("Mixtral", "mixtral:8x22b", "registry"),
    "2": ("EngGPT", "enggpt", os.getenv("ENGGPT_REPO", "engineering-group/EngGPT2-16B-A3B")),
    "3": ("Llama 3.2 3B", "llama3.2:3b", "registry"),
    "4": ("qwen2.5:7b", "qwen2.5:7b", "registry"),
}

# ENGGPT_DIR = os.getenv("ENGGPT_DIR", "./enggpt")

ollama_client = Client(host=OLLAMA_HOST)
