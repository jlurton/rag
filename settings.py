"""
Paramètres partagés par tous les TP et par `rag.py`.

Un seul endroit à modifier pour changer de modèle, de taille de chunk, etc.
"""

# --- Chemins ---------------------------------------------------------------
DATA_DIR = "data"  # les PDF (voir download_data.py)
INDEX_DIR = "index"  # l'index calculé : chunks.json + embeddings.npy

# --- Découpage des documents (TP1) ----------------------------------------
CHUNK_SIZE_WORDS = 150  # taille d'un chunk, en mots
CHUNK_OVERLAP_WORDS = 30  # chevauchement entre deux chunks consécutifs

# --- Modèle d'embeddings (TP1) --------------------------------------------
# Modèle ouvert Hugging Face, exécuté en local sur le CPU (via fastembed).
# ATTENTION : l'index doit être recalculé si on change de modèle.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modèles comparés dans le TP1 (dimensions, langues et tailles différentes)
EMBEDDING_MODELS_TO_COMPARE = [
    "sentence-transformers/all-MiniLM-L6-v2",  # petit, anglais
    "BAAI/bge-base-en-v1.5",  # plus gros, anglais
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # multilingue (le modèle du TP)
]

# --- Modèle de génération (TP2) -------------------------------------------
# Appelé via le proxy LLM de Cross Data (API compatible OpenAI), avec LiteLLM :
# le préfixe "openai/" indique à LiteLLM de parler le format OpenAI à l'adresse LLM_API_BASE.
# La clé d'API est lue dans la variable d'environnement LLM_API_KEY (fichier .env).
LLM_API_BASE = "https://proxy.openllm.lab.crossdata.tech"
GENERATION_MODEL = "openai/mistral/mistral-small-latest"
# Modèle utilisé pour les comparaisons (TP2 §4.b) et comme juge (TP3).
# Le proxy n'expose qu'un seul modèle de génération pour l'instant : c'est donc le même.
# À remplacer par un modèle plus gros dès qu'il y en a un.
GENERATION_MODEL_LARGE = "openai/mistral/mistral-small-latest"

# --- Recherche -------------------------------------------------------------
TOP_K = 5  # nombre de chunks transmis au LLM
