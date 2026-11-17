"""
Corrigé du TP3 — les fonctions d'évaluation et des bonus, prêtes à importer.

    from solutions.tp3 import evaluer_retrieval, hit_at_k, mrr, afficher_resultats
    from solutions.tp3 import refuse, refuse_llm
"""

import matplotlib.pyplot as plt
import numpy as np

from rag import generate


# --- Partie 2 — évaluer la recherche avec le hit@k ----------------------------
def rang_bonne_page(item, resultats_recherche):
    """Rang (à partir de 1) du premier chunk issu de la bonne page ; None si elle est absente."""
    for rang, r in enumerate(resultats_recherche, start=1):
        if r["source"] == item["source"] and r["page"] in item["pages"]:
            return rang
    return None


def evaluer_retrieval(fonction_recherche, jeu_test, k_max=10):
    """Lance la recherche sur chaque question et renvoie la liste des rangs de la bonne page.

    fonction_recherche(question, k) -> liste de chunks
    """
    return [rang_bonne_page(item, fonction_recherche(item["question"], k_max)) for item in jeu_test]


def hit_at_k(rangs, k):
    """Proportion de questions dont la bonne page est dans les k premiers résultats."""
    return float(np.mean([rang is not None and rang <= k for rang in rangs]))


# --- Bonus 1 — le MRR ---------------------------------------------------------
def mrr(rangs):
    """Moyenne de 1/rang de la bonne page (0 quand elle est absente)."""
    return float(np.mean([1 / rang if rang else 0 for rang in rangs]))


def afficher_resultats(resultats):
    """Tableau (hit@1, hit@5, hit@10, MRR) et courbes hit@k de toutes les variantes (resultats = {nom: rangs})."""
    for nom, rangs_variante in resultats.items():
        print(f"{nom:<32} hit@1={hit_at_k(rangs_variante, 1):.2f}  hit@5={hit_at_k(rangs_variante, 5):.2f}  "
              f"hit@10={hit_at_k(rangs_variante, 10):.2f}  MRR={mrr(rangs_variante):.2f}")
        plt.plot(range(1, 11), [hit_at_k(rangs_variante, k) for k in range(1, 11)], marker="o", label=nom)
    plt.xlabel("k"); plt.ylabel("hit@k"); plt.ylim(0, 1.05); plt.grid(alpha=0.3); plt.legend()
    plt.title("La bonne page est-elle dans les k premiers résultats ?")
    plt.show()


# --- Bonus « questions pièges » : détecter un refus ---------------------------
def refuse(reponse):
    """Option 1 : mots-clés. Simple et gratuit, mais fragile : il suffit d'une formulation imprévue."""
    indices = ["ne sais pas", "ne figure pas", "pas dans les extraits", "ne permettent pas",
               "aucune information", "n'est pas mentionn", "information absente"]
    return any(indice in reponse.lower() for indice in indices)


def refuse_llm(reponse):
    """Option 2 : demander au LLM. Plus fiable, mais un appel (donc un coût) de plus par question."""
    messages = [{"role": "user", "content": "Ce texte dit-il que l'information demandée n'est pas disponible ? "
                 f"Réponds uniquement par OUI ou NON.\n\n---\n{reponse}"}]
    return "OUI" in generate(messages).upper()
