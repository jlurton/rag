"""
Corrigé du TP2 — les fonctions des exercices, prêtes à importer.

    from solutions.tp2 import SYSTEM_PROMPT, rewrite_query, chat
"""

import settings
from rag import build_prompt, generate, search

# --- Exercice 1 — le prompt système (une solution parmi d'autres) -------------
# La phrase de refus standardisée rend les refus faciles à détecter (TP3, bonus « questions pièges »).
SYSTEM_PROMPT = """Tu réponds à des questions techniques à partir d'extraits de datasheets.
Règles :
- Donne d'abord la valeur demandée avec son unité, puis une phrase d'explication au maximum.
- Utilise UNIQUEMENT les extraits fournis. Si l'information n'y est pas, réponds exactement : "Information absente des extraits fournis."
- Termine par la source : [fichier, page N].
- Réponds dans la langue de la question."""


# --- Exercice 2 — la recherche hybride dans le RAG ----------------------------
def answer_hybride(question, chunks, embeddings, k=settings.TOP_K):
    """Comme rag.answer, mais avec la recherche hybride du TP1."""
    from solutions.tp1 import search_hybrid

    extraits = search_hybrid(question, chunks, embeddings, k)
    return generate(build_prompt(question, extraits)), extraits


# --- Exercice 3 — réécrire la question avant la recherche ---------------------
def rewrite_query(question):
    """Demande au LLM une requête courte, en anglais technique, proche du vocabulaire des datasheets."""
    messages = [
        {"role": "system", "content": "Transforme la question en une requête de recherche courte, en anglais "
         "technique, avec les termes exacts qu'on trouverait dans une datasheet de composant électronique. "
         "Réponds uniquement par la requête, sans guillemets ni commentaire."},
        {"role": "user", "content": question},
    ]
    return generate(messages)


# --- Bonus — un RAG conversationnel ------------------------------------------
def chat(message, historique, chunks, embeddings, k=settings.TOP_K):
    """Ajoute un tour à la discussion et renvoie la réponse.

    `historique` est la liste des messages échangés ; on la renvoie entière au modèle à chaque tour.
    Exemple : historique = [{"role": "system", "content": SYSTEM_PROMPT}]
    """
    extraits = search(message, chunks, embeddings, k)
    extraits_texte = "\n\n".join(f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in extraits)
    historique.append({"role": "user", "content": f"Extraits de documentation :\n\n{extraits_texte}\n\nQuestion : {message}"})
    reponse = generate(historique)
    historique.append({"role": "assistant", "content": reponse})
    return reponse
