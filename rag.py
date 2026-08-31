"""
rag.py — un RAG complet, volontairement simple, en un seul fichier.

Un RAG (Retrieval-Augmented Generation) enchaîne deux grandes phases :

  A. INDEXATION (une seule fois, hors ligne)
     1. lire les documents                       -> load_pdfs()
     2. les découper en morceaux ("chunks")       -> chunk_pages()
     3. vectoriser chaque chunk (embeddings)      -> embed()
     4. sauvegarder chunks + vecteurs             -> build_index() / load_index()

  B. RÉPONSE (à chaque question)
     5. retrouver les chunks pertinents           -> search()        (retrieval)
     6. générer la réponse à partir de ces chunks -> generate()      (generation)
     7. 5 + 6 = le RAG                            -> answer()

Usage en ligne de commande :
    python rag.py                      # construit l'index (data/ -> index/)
    python rag.py "Quelle est la tension d'alimentation du HC-SR04 ?"

Les notebooks tp1/tp2/tp3 reprennent ces étapes une par une, avec des explications.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pymupdf  # lecture des PDF
from dotenv import load_dotenv
from fastembed import TextEmbedding  # modèles d'embeddings ouverts, exécutés en local
from litellm import completion  # appel des LLMs via API
from rank_bm25 import BM25Okapi  # recherche par mots-clés

import settings

load_dotenv()  # charge HF_TOKEN depuis le fichier .env
warnings.filterwarnings("ignore", message=".*mean pooling.*")  # avertissement interne de fastembed, sans conséquence


# =============================================================================
# A. INDEXATION
# =============================================================================


# --- 1. Lire les documents ---------------------------------------------------
def load_pdfs(data_dir: str = settings.DATA_DIR) -> list[dict]:
    """Lit tous les PDF d'un dossier et renvoie une liste de pages.

    Une page = {"source": "esp32.pdf", "page": 12, "text": "..."}.
    On garde la source et le numéro de page : ils serviront à citer les sources.
    """
    pages = []
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        document = pymupdf.open(pdf_path)
        for number, page in enumerate(document, start=1):
            text = page.get_text()
            if text.strip():  # on ignore les pages vides (images seules)
                pages.append({"source": pdf_path.name, "page": number, "text": text})
    return pages


# --- 2. Découper en chunks ---------------------------------------------------
def chunk_pages(
    pages: list[dict],
    chunk_size: int = settings.CHUNK_SIZE_WORDS,
    overlap: int = settings.CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """Découpe chaque page en chunks de `chunk_size` mots, avec chevauchement.

    C'est la méthode la plus naïve possible : on coupe tous les N mots, sans
    tenir compte des phrases, des titres ni des tableaux. À améliorer (TP1) !
    """
    chunks = []
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[start : start + chunk_size])
            chunks.append(
                {
                    "id": len(chunks),
                    "source": page["source"],
                    "page": page["page"],
                    "text": chunk_text,
                }
            )
    return chunks


# --- 3. Vectoriser (embeddings) ----------------------------------------------
_embedding_model = None  # chargé une seule fois


def embed(texts: list[str], model_name: str = settings.EMBEDDING_MODEL) -> np.ndarray:
    """Transforme une liste de textes en une matrice de vecteurs (n_textes x dim).

    Les vecteurs sont normalisés (norme = 1) : la similarité cosinus entre deux
    textes devient alors un simple produit scalaire.
    """
    global _embedding_model
    if _embedding_model is None or _embedding_model.model_name != model_name:
        _embedding_model = TextEmbedding(model_name)  # téléchargé au 1er appel
    vectors = np.array(list(_embedding_model.embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# --- 4. Construire / charger l'index ------------------------------------------
def build_index(data_dir: str = settings.DATA_DIR, index_dir: str = settings.INDEX_DIR) -> None:
    """Enchaîne les étapes 1 à 3 et sauvegarde le résultat sur disque.

    L'index tient en deux fichiers :
      - chunks.json    : la liste des chunks (texte + source + page)
      - embeddings.npy : la matrice des vecteurs, une ligne par chunk
    """
    print("1. lecture des PDF...")
    pages = load_pdfs(data_dir)
    print(f"   {len(pages)} pages lues")

    print("2. découpage en chunks...")
    chunks = chunk_pages(pages)
    print(f"   {len(chunks)} chunks")

    print(f"3. calcul des embeddings avec {settings.EMBEDDING_MODEL} (peut prendre 1 à 2 min)...")
    embeddings = embed([chunk["text"] for chunk in chunks])
    print(f"   matrice {embeddings.shape} (chunks x dimensions)")

    print("4. sauvegarde de l'index...")
    save_index(chunks, embeddings, index_dir)
    print(f"   index écrit dans {index_dir}/")


def save_index(chunks: list[dict], embeddings: np.ndarray, index_dir: str) -> None:
    """Sauvegarde un index (chunks + embeddings) dans un dossier.

    Permet de conserver plusieurs index côte à côte (par exemple un par stratégie
    de découpage) et de les recharger avec load_index(index_dir) pour les comparer.
    """
    Path(index_dir).mkdir(exist_ok=True)
    with open(Path(index_dir) / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    np.save(Path(index_dir) / "embeddings.npy", embeddings)


def load_index(index_dir: str = settings.INDEX_DIR) -> tuple[list[dict], np.ndarray]:
    """Recharge l'index depuis le disque (évite de tout recalculer)."""
    with open(Path(index_dir) / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(Path(index_dir) / "embeddings.npy")
    return chunks, embeddings


# =============================================================================
# B. RÉPONSE
# =============================================================================


# --- 5. Retrieval : retrouver les chunks pertinents ---------------------------
def search(query: str, chunks: list[dict], embeddings: np.ndarray, k: int = settings.TOP_K) -> list[dict]:
    """Recherche sémantique : les k chunks dont le vecteur est le plus proche de la question.

    1. on vectorise la question avec le MÊME modèle que les chunks
    2. similarité cosinus avec tous les chunks = un produit matrice-vecteur
    3. on garde les k meilleurs scores
    """
    query_vector = embed([query])[0]
    scores = embeddings @ query_vector  # un score par chunk, entre -1 et 1
    best = np.argsort(scores)[::-1][:k]  # indices des k plus grands scores
    return [{**chunks[i], "score": float(scores[i])} for i in best]


def tokenize(text: str) -> list[str]:
    """Découpage très simple en mots, pour BM25 (minuscules, sans ponctuation)."""
    return re.findall(r"\w+", text.lower())


def search_bm25(query: str, chunks: list[dict], k: int = settings.TOP_K) -> list[dict]:
    """Recherche par mots-clés (BM25) : l'approche classique des moteurs de recherche.

    Complémentaire de la recherche sémantique : très bonne sur les termes exacts
    (références de composants, noms de broches, sigles...), aveugle aux synonymes.
    """
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
    scores = bm25.get_scores(tokenize(query))
    best = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]


# --- 6. Generation : répondre à partir des chunks ----------------------------
SYSTEM_PROMPT = """Tu es un assistant technique qui répond à des questions sur des datasheets de composants électroniques.
Réponds UNIQUEMENT à partir des extraits fournis. Si l'information n'y figure pas, dis-le clairement.
Cite tes sources à la fin de chaque affirmation sous la forme [fichier, page N].
Réponds dans la langue de la question."""


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Assemble le prompt : consignes (system) + extraits + question (user).

    C'est le coeur du "augmented" : la connaissance n'est pas dans le modèle,
    elle est injectée dans le prompt.
    """
    extraits = "\n\n".join(
        f"[{chunk['source']}, page {chunk['page']}]\n{chunk['text']}" for chunk in chunks
    )
    user_message = f"Extraits de documentation :\n\n{extraits}\n\nQuestion : {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate(messages: list[dict], model: str = settings.GENERATION_MODEL, temperature: float = 0.0) -> str:
    """Appelle le LLM via LiteLLM et renvoie le texte de la réponse.

    temperature=0 : réponses les plus déterministes possible (ce qu'on veut pour du factuel).
    """
    response = completion(model=model, messages=messages, temperature=temperature, max_tokens=600)
    return response.choices[0].message.content


# --- 7. Le RAG = retrieval + generation --------------------------------------
def answer(query: str, chunks: list[dict], embeddings: np.ndarray, k: int = settings.TOP_K) -> tuple[str, list[dict]]:
    """Répond à une question : recherche des k chunks pertinents puis génération.

    Renvoie la réponse ET les chunks utilisés (pour pouvoir vérifier les sources).
    """
    relevant_chunks = search(query, chunks, embeddings, k)
    messages = build_prompt(query, relevant_chunks)
    return generate(messages), relevant_chunks


# =============================================================================
# Ligne de commande
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) == 1:
        build_index()
    else:
        question = " ".join(sys.argv[1:])
        chunks, embeddings = load_index()
        response, used_chunks = answer(question, chunks, embeddings)
        print(response)
        print("\n--- chunks utilisés ---")
        for chunk in used_chunks:
            print(f"[{chunk['source']}, page {chunk['page']}] score={chunk['score']:.3f}")
