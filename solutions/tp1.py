"""
Corrigé du TP1 — les fonctions des exercices, prêtes à importer.

Fourni en fin de TP1 : si vous n'avez pas terminé les exercices, les TP2 et TP3
importent ces fonctions pour que vous puissiez continuer.

    from solutions.tp1 import search_hybrid, chunk_pages_par_paragraphes
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag import search

# --- Exercice 1 a) les mots vides ---------------------------------------------
MOTS_VIDES = set(
    "le la les l un une des du de d et ou à au aux en dans par pour sur avec sans est sont a ont que qui quoi "
    "quel quelle quels quelles comment combien ce cet cette ces se sa son ses il elle on ne pas plus "
    "the a an of to in on for and or is are be by with at from as that this which what how many much it its".split()
)


def tokenize_sans_mots_vides(texte):
    """Comme rag.tokenize, mais sans les mots vides (le, de, the, of...)."""
    return [mot for mot in re.findall(r"\w+", texte.lower()) if mot not in MOTS_VIDES]


def search_bm25_sans_mots_vides(question, chunks, k=5):
    """Comme rag.search_bm25, avec le découpage en mots sans mots vides."""
    bm25 = BM25Okapi([tokenize_sans_mots_vides(c["text"]) for c in chunks])
    scores = bm25.get_scores(tokenize_sans_mots_vides(question))
    meilleurs = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in meilleurs]


# --- Exercice 1 b) la fusion (Reciprocal Rank Fusion) ------------------------
def search_hybrid(question, chunks, embeddings, k=5, n=20):
    """Recherche hybride : fusionne les rangs de la recherche sémantique et de BM25.

    Chaque chunk reçoit, sur chacune des deux listes, 1 / (60 + rang).
    Un chunk bien classé dans les deux listes remonte en tête.
    """
    scores = {}
    for liste in [search(question, chunks, embeddings, k=n), search_bm25_sans_mots_vides(question, chunks, k=n)]:
        for rang, chunk in enumerate(liste, start=1):
            scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (60 + rang)
    meilleurs = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**chunks[i], "score": scores[i]} for i in meilleurs]


# --- Exercice 2 — découper par paragraphes -----------------------------------
def chunk_pages_par_paragraphes(pages, max_words=150):
    """Regroupe les paragraphes consécutifs d'une page jusqu'à max_words mots.

    Chaque chunk commence par le nom du document, par exemple "[HCSR04] ...".
    """
    chunks_resultat = []

    def ajouter(page, prefixe, mots):
        chunks_resultat.append({"id": len(chunks_resultat), "source": page["source"],
                                "page": page["page"], "text": prefixe + " ".join(mots)})

    for page in pages:
        prefixe = f"[{page['source'].removesuffix('.pdf').upper()}] "
        courant = []
        for paragraphe in re.split(r"\n\s*\n", page["text"]):
            mots = paragraphe.split()
            if not mots:
                continue
            # le paragraphe ne rentre plus dans le chunk courant : on ferme le chunk
            if courant and len(courant) + len(mots) > max_words:
                ajouter(page, prefixe, courant)
                courant = []
            # paragraphe trop long pour un seul chunk : on le coupe
            while len(mots) > max_words:
                ajouter(page, prefixe, mots[:max_words])
                mots = mots[max_words:]
            courant += mots
        if courant:
            ajouter(page, prefixe, courant)
    return chunks_resultat
