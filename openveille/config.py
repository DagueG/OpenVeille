# -*- coding: utf-8 -*-
"""Configuration centralisée : env vars et constantes du pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://openveilleapi.openai.azure.com/")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")

# BOAMP
BOAMP_URL = "https://www.boamp.fr/api/explore/v2.1/catalog/datasets/boamp/records"

# Pipeline defaults (déterminés lors du test embeddings + rerank)
DEFAULT_TOP_K_RERANK = 30       # nb d'AO envoyés au LLM par profil
DEFAULT_NOTIF_THRESHOLD = 70    # seuil LLM pour marquer pépite (validé sur golden set)
MIN_DESC_LENGTH = 400           # skip AO squelette

# Concurrence
EMBED_BATCH_SIZE = 100
LLM_MAX_WORKERS = 5             # parallélisme du re-rank LLM

# Chemins
OUT_DIR = Path("out")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")