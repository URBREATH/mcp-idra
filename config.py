# -*- coding: utf-8 -*-
"""Configurazione centralizzata e client Ollama condiviso."""

import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama-backend:11434")
PILOT_CITIES = "Aarhus,Athens,Cluj-Napoca,Kajaani,Leuven,Madrid,Parma,Pilsen,Tallinn"
IDRA_TAG = "IDRA"

MODELS = {
    "1": ("Mixtral", "mixtral:8x22b", "registry"),
    "2": ("EngGPT", "enggpt", os.getenv("ENGGPT_REPO", "engineering-group/EngGPT2-16B-A3B")),
    "3": ("TildeOpen-30b", "TildeOpen-30b", os.getenv("TILDE_REPO", "TildeAI/TildeOpen-30b"))
}

ollama_client = Client(host=OLLAMA_HOST)
