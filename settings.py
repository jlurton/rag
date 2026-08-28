"""
Paramètres partagés par tous les TP et par `rag.py`.

Un seul endroit à modifier pour changer de modèle, de taille de chunk, etc.
"""

# --- Chemins ---------------------------------------------------------------
DATA_DIR = "data"    # les PDF (voir download_data.py)
INDEX_DIR = "index"  # l'index calculé : chunks.json + embeddings.npy

# --- Découpage des documents (TP1) ----------------------------------------
CHUNK_SIZE_WORDS = 150   # taille d'un chunk, en mots
CHUNK_OVERLAP_WORDS = 30 # chevauchement entre deux chunks consécutifs

# --- Modèle d'embeddings (TP1) --------------------------------------------
# Modèle ouvert Hugging Face, exécuté en local sur le CPU (via fastembed).
# ATTENTION : l'index doit être recalculé si on change de modèle.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Modèle de génération (TP2) -------------------------------------------
# Appelé via l'API Hugging Face Inference Providers, avec LiteLLM.
# Syntaxe LiteLLM : "huggingface/<organisation>/<modèle>".
# La clé d'API est lue dans la variable d'environnement HF_TOKEN (fichier .env).
GENERATION_MODEL = "huggingface/Qwen/Qwen2.5-7B-Instruct"
# Un modèle beaucoup plus gros, pour comparer coût / qualité :
GENERATION_MODEL_LARGE = "huggingface/Qwen/Qwen2.5-72B-Instruct"

# --- Recherche -------------------------------------------------------------
TOP_K = 5  # nombre de chunks transmis au LLM
