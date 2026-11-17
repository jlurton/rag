"""
Corrigé du TP4 — les fonctions des trois parties, prêtes à importer.

    from solutions.tp4 import rerank, search_rerank, document_hypothetique, search_hyde, enrichir_chunks
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder

import settings
from rag import generate, search

# --- Partie 1 — le reranking --------------------------------------------------
_reranker = None  # chargé une seule fois (téléchargé au premier appel, ~80 Mo)


def rerank(question, candidats, k=5):
    """Reclasse les candidats avec le cross-encoder et garde les k meilleurs."""
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(settings.RERANK_MODEL)
    scores = list(_reranker.rerank(question, [c["text"] for c in candidats]))
    meilleurs = np.argsort(scores)[::-1][:k]
    return [{**candidats[i], "score": float(scores[i])} for i in meilleurs]


def search_rerank(question, chunks, embeddings, k=5, n=20):
    """Deux étages : la recherche sémantique ramène n candidats, le cross-encoder en garde k."""
    return rerank(question, search(question, chunks, embeddings, k=n), k)


# --- Partie 2 — HyDE ----------------------------------------------------------
PROMPT_HYDE = """Écris le court passage d'une datasheet de composant électronique qui répondrait à cette question.
Écris-le en anglais, dans le style d'une datasheet : termes techniques, valeurs, unités, 3 à 5 lignes.
Si tu ne connais pas la valeur exacte, donne une valeur plausible. Réponds uniquement par le passage.

Question : {question}"""


def document_hypothetique(question):
    """Demande au LLM le passage qu'il imagine trouver dans la datasheet."""
    return generate([{"role": "user", "content": PROMPT_HYDE.format(question=question)}])


def search_hyde(question, chunks, embeddings, k=5):
    """Cherche avec le passage hypothétique au lieu de la question."""
    return search(document_hypothetique(question), chunks, embeddings, k)


# --- Partie 3 — enrichir les chunks -------------------------------------------
PROMPT_CONTEXTE = """Voici une page de la datasheet {source} :
---
{page}
---

Voici un extrait de cette page :
---
{extrait}
---

Écris une seule phrase, en français, qui situe cet extrait : le composant, la partie du document et le sujet traité.
Cette phrase servira à retrouver l'extrait lors d'une recherche. Réponds uniquement par la phrase."""


def contexte_chunk(chunk, texte_page):
    """Une phrase de contexte rédigée par le LLM, qui situe le chunk dans son document."""
    prompt = PROMPT_CONTEXTE.format(source=chunk["source"], page=texte_page[:3000], extrait=chunk["text"])
    return generate([{"role": "user", "content": prompt}]).strip()


def enrichir_chunks(chunks, pages, fonction_contexte=contexte_chunk, appels_simultanes=8):
    """Ajoute à chaque chunk sa phrase de contexte, en tête du texte.

    Les appels au LLM partent `appels_simultanes` par `appels_simultanes` : sinon, 600 appels l'un après
    l'autre prendraient 10 à 20 minutes.
    """
    texte_page = {(p["source"], p["page"]): p["text"] for p in pages}

    def enrichir(chunk):
        try:
            contexte = fonction_contexte(chunk, texte_page[(chunk["source"], chunk["page"])])
        except Exception as erreur:  # un appel en échec ne doit pas faire perdre les 600 autres
            print(f"chunk {chunk['id']} : appel en échec ({type(erreur).__name__}), chunk gardé sans contexte")
            contexte = ""
        return {**chunk, "contexte": contexte, "text": f"{contexte}\n{chunk['text']}".strip()}

    with ThreadPoolExecutor(max_workers=appels_simultanes) as executor:
        chunks_enrichis = list(executor.map(enrichir, chunks))
    sans_contexte = sum(not c["contexte"] for c in chunks_enrichis)
    if sans_contexte:
        print(f"{sans_contexte} chunks sur {len(chunks)} sans contexte : vérifiez la clé d'API et relancez.")
    return chunks_enrichis
