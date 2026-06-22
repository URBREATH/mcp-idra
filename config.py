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
    "3": ("TildeOpen-30b", "TildeOpen-30b", os.getenv("TILDE_REPO", "TildeAI/TildeOpen-30b"))
}

# ENGGPT_DIR = os.getenv("ENGGPT_DIR", "./enggpt")

ollama_client = Client(host=OLLAMA_HOST)
