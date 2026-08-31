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
# Appelé via l'API Hugging Face Inference Providers, avec LiteLLM.
# Syntaxe LiteLLM : "huggingface/<organisation>/<modèle>".
# La clé d'API est lue dans la variable d'environnement HF_TOKEN (fichier .env).
# Petit modèle (~4 milliards de paramètres), multilingue, ~0,01 $/M tokens.
GENERATION_MODEL = "huggingface/Qwen/Qwen3-4B-Instruct-2507"
# Un modèle 18 fois plus gros (~0,40 $/M tokens), pour comparer coût / qualité (TP2) et juger (TP3) :
GENERATION_MODEL_LARGE = "huggingface/Qwen/Qwen2.5-72B-Instruct"

# --- Recherche -------------------------------------------------------------
TOP_K = 5  # nombre de chunks transmis au LLM
