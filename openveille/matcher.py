# -*- coding: utf-8 -*-
"""Pipeline de matching 2 étages : embedding cosinus → LLM re-rank."""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import AzureOpenAI

from .config import (
    AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_API_KEY,
    EMBED_DEPLOYMENT, CHAT_DEPLOYMENT,
    EMBED_BATCH_SIZE, LLM_MAX_WORKERS,
)

log = logging.getLogger(__name__)

_client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """
    Embed en batches. Retourne (n, d) normalisée L2 → produit scalaire = cosinus.
    Azure text-embedding-3-small : 1536 dims.
    Petite pause entre batchs pour rester sous le quota TPM.
    """
    all_vecs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        i = bi * batch_size
        batch = texts[i:i + batch_size]
        t0 = time.time()
        resp = _client.embeddings.create(model=EMBED_DEPLOYMENT, input=batch)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        all_vecs.append(vecs)
        log.info("  Embed batch [%d..%d[ (%d textes) en %.1fs",
                 i, i + len(batch), len(batch), time.time() - t0)
        if bi < n_batches - 1:
            time.sleep(1.0)  # anti-429 : reste sous le TPM Azure
    vecs = np.vstack(all_vecs)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-10, None)

def rank_by_similarity(profile_vec: np.ndarray,
                       ao_vecs: np.ndarray,
                       aos: list[dict],
                       top_k: int) -> list[dict]:
    """
    Trie les AO par similarité cosinus au profil. Retourne les top_k enrichis
    du champ `embedding_sim` et `embedding_rank`.
    profile_vec : shape (1, d) ou (d,)
    ao_vecs     : shape (n, d)
    """
    p = profile_vec.reshape(1, -1) if profile_vec.ndim == 1 else profile_vec
    sims = (p @ ao_vecs.T).flatten()

    idx_sorted = np.argsort(-sims)  # descendant
    top_idx = idx_sorted[:top_k]

    ranked = []
    for rank, i in enumerate(top_idx, 1):
        ranked.append({
            **aos[int(i)],
            "embedding_sim": float(sims[int(i)]),
            "embedding_rank": rank,
        })
    return ranked


_PROMPT_SYSTEM = """Tu es un expert en veille des appels d'offres publics français.
Ta mission : identifier UNIQUEMENT les vraies pépites pour une PME donnée,
c'est-à-dire des marchés où la PME a une chance réaliste de gagner ET pas juste
"des compétences transposables" ou "un domaine proche".

Tu réponds UNIQUEMENT en JSON valide avec cette structure exacte :
{"score": <entier 0-100>, "reason": "<explication en français, 200 caractères max>"}

Barème STRICT :
- 90-100 : cœur de métier exact — la PME est le prestataire idéal, prestation demandée = activité principale.
- 80-89  : match direct — les prestations demandées relèvent d'une des activités listées dans le profil, la PME est un candidat naturel avec de bonnes chances.
- 40-79  : adjacent, périphérique, transposable — la PME "pourrait éventuellement" candidater mais ce n'est PAS son métier. NE PAS notifier.
- 0-39   : hors sujet total.

Règles ANTI-généreux (à appliquer avant de valider un score ≥ 80) :

1. Similitude technique ≠ match métier.
   Une ESN gestion documentaire qui voit un marché "Talend/BI" → 60-70, pas 80+.
   Un cabinet conseil qui voit un marché "logiciel de gestion" → 40-60, pas 70+.
   Une entreprise rénovation énergétique qui voit "bardage bois" → 40-60, pas 70+.

2. Maintenance/exploitation ≠ travaux neufs.
   Si le profil décrit une entreprise de travaux/conception et l'AO parle
   d'exploitation/maintenance/entretien : score max 65, même si le domaine matche.

3. Le profil doit correspondre au TYPE de prestataire attendu.
   BE thermique ≠ entreprise BTP. ESN ≠ cabinet conseil. Fournisseur ≠ intégrateur.
   Si l'AO cherche clairement un autre type d'acteur : score max 60.

4. Test final avant de valider ≥ 80 :
   "Le business developer de la PME investirait-il 2 jours à monter un dossier
   pour ce marché ?" Si tu hésites, le score est < 80.

Sois EXIGEANT. La promesse produit est "les vraies pépites, pas le bruit adjacent".
Il vaut mille fois mieux dire "0 pépite cette semaine" que de proposer 5 marchés
tangents. L'utilisateur préfère un service qui se tait quand il n'y a rien
plutôt qu'un service qui invente des matches."""

def _rerank_one(profile_text: str, ao_objet: str, ao_desc: str) -> tuple[int, str, str]:
    """Retourne (score, reason, error). error vide si OK."""
    try:
        resp = _client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT_SYSTEM},
                {"role": "user", "content": (
                    f"=== PROFIL DE LA PME ===\n{profile_text.strip()}\n\n"
                    f"=== APPEL D'OFFRES ===\nTitre : {ao_objet}\n"
                    f"Description : {ao_desc}\n\n"
                    "Évalue cet AO pour cette PME et réponds en JSON."
                )},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        score = max(0, min(100, int(parsed.get("score", 0))))
        reason = str(parsed.get("reason", ""))[:250]
        return score, reason, ""
    except Exception as e:
        return 0, "", f"{type(e).__name__}: {e}"


def rerank_llm(profile_text: str,
               ranked_aos: list[dict],
               max_workers: int = LLM_MAX_WORKERS) -> list[dict]:
    """
    LLM re-rank en parallèle (ThreadPoolExecutor). Enrichit chaque AO avec
    llm_score, llm_reason, llm_error, puis retrie par llm_score desc.
    """
    log.info("LLM re-rank sur %d AO (parallèle, %d workers)...",
             len(ranked_aos), max_workers)
    t_start = time.time()

    def _process(ao):
        score, reason, err = _rerank_one(
            profile_text, ao["objet"], ao.get("description", "")[:1500]
        )
        return {**ao, "llm_score": score, "llm_reason": reason, "llm_error": err}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        out = list(ex.map(_process, ranked_aos))

    n_err = sum(1 for a in out if a["llm_error"])
    log.info("LLM re-rank terminé en %.1fs (%d erreurs)",
             time.time() - t_start, n_err)

    out.sort(key=lambda x: x["llm_score"], reverse=True)
    for rank, ao in enumerate(out, 1):
        ao["llm_rank"] = rank
    return out

def rerank_llm_from_db_candidates(profile_text: str,
                                   candidates: list[dict],
                                   max_workers: int = LLM_MAX_WORKERS) -> list[dict]:
    """
    Re-rank LLM sur des candidats venant de la RPC match_ao_by_embedding.
    Format d'entrée : dicts avec idweb, objet, description, similarity, etc.
    Sortie triée par llm_score desc.
    """
    log.info("LLM re-rank sur %d candidats DB (parallèle, %d workers)...",
             len(candidates), max_workers)
    t_start = time.time()

    def _process(cand):
        score, reason, err = _rerank_one(
            profile_text, cand["objet"], (cand.get("description") or "")[:1500]
        )
        return {**cand, "llm_score": score, "llm_reason": reason, "llm_error": err}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        out = list(ex.map(_process, candidates))

    n_err = sum(1 for a in out if a["llm_error"])
    log.info("LLM re-rank terminé en %.1fs (%d erreurs)",
             time.time() - t_start, n_err)

    out.sort(key=lambda x: x["llm_score"], reverse=True)
    for rank, ao in enumerate(out, 1):
        ao["llm_rank"] = rank
    return out