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
    "0": ("TildeOpen", "TildeAI/TildeOpen-30b-64k", "TildeAI/TildeOpen-30b-64k", "transformers"),
    "1": ("Mixtral 8x7B Instruct", "mixtral:instruct", "registry", "ollama"),
    "2": ("EngGPT2 16B-A3B",       "engineering-group/EngGPT2-16B-A3B",
                                    "engineering-group/EngGPT2-16B-A3B", "transformers")
}

ollama_client = Client(host=OLLAMA_HOST)
