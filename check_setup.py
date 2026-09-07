"""
Vérifie que l'environnement est prêt pour le TP.

Usage : poetry run python check_setup.py

Vérifie dans l'ordre :
  1. la version de Python
  2. la clé d'API (fichier .env)
  3. le corpus (dossier data/)
  4. le modèle d'embeddings (téléchargé au premier appel, ~500 Mo)
  5. un appel au LLM via l'API
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import settings

OK, KO = "  [OK] ", "  [KO] "

print("1. Python")
if sys.version_info >= (3, 11):
    print(OK + sys.version.split()[0])
else:
    sys.exit(KO + f"Python {sys.version.split()[0]} : il faut Python >= 3.11")

print("2. Clé d'API")
load_dotenv()
if os.getenv("LLM_API_KEY"):
    print(OK + "LLM_API_KEY trouvée dans .env")
else:
    sys.exit(KO + "LLM_API_KEY absente : copier .env.example en .env et renseigner la clé fournie")

print("3. Corpus")
pdfs = list(Path(settings.DATA_DIR).glob("*.pdf"))
if pdfs:
    print(OK + f"{len(pdfs)} PDF dans {settings.DATA_DIR}/")
else:
    sys.exit(KO + f"aucun PDF dans {settings.DATA_DIR}/ : lancer `python download_data.py`")

print("4. Modèle d'embeddings (téléchargement au premier lancement, patience...)")
from rag import embed  # noqa: E402  (import ici pour que les étapes 1-3 restent rapides)

vector = embed(["test"])[0]
print(OK + f"{settings.EMBEDDING_MODEL} -> vecteur de dimension {len(vector)}")

print("5. Appel du LLM via l'API")
from rag import generate  # noqa: E402

reply = generate([{"role": "user", "content": "Réponds uniquement par le mot : OK"}])
print(OK + f"{settings.GENERATION_MODEL} a répondu : {reply.strip()!r}")

print("\nTout est prêt !")
