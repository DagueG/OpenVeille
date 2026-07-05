# -*- coding: utf-8 -*-
"""
Persistance Supabase : boamp_ao, profile, match.

Design :
- Upsert idempotent sur idweb (pas de doublons si on re-run)
- Embeddings stockés une fois pour toutes (économie de tokens Azure)
- Recherche vectorielle côté SQL via RPC pgvector
"""
import logging
from typing import Iterable

import numpy as np
from supabase import create_client, Client

from .config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

log = logging.getLogger(__name__)

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY requis dans .env"
    )

_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_client() -> Client:
    return _client


# ---------- BOAMP AO ----------

def _vec_to_pg(vec: np.ndarray | list[float] | None) -> list[float] | None:
    """Convertit un vecteur numpy en liste Python (format attendu par pgvector via REST)."""
    if vec is None:
        return None
    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return list(vec)


def upsert_boamp_ao_batch(aos: list[dict], embeddings: np.ndarray | None = None,
                          batch_size: int = 50) -> int:
    """
    Upsert un batch d'AO. Si embeddings est fourni (shape (n, 1536)), il est stocké.
    Idempotent sur idweb.
    """
    if embeddings is not None and len(embeddings) != len(aos):
        raise ValueError(
            f"len(embeddings)={len(embeddings)} != len(aos)={len(aos)}"
        )

    rows = []
    for i, ao in enumerate(aos):
        emb = _vec_to_pg(embeddings[i]) if embeddings is not None else None
        # Certains champs BOAMP arrivent en liste (ex: descripteur_code)
        row = {
            "idweb": ao["idweb"],
            "dateparution": ao.get("dateparution"),
            "datelimitereponse": ao.get("datelimitereponse"),
            "objet": ao["objet"],
            "description": ao["description"],
            "acheteur": _as_scalar_str(ao.get("acheteur")),
            "code_departement": _as_scalar_str(ao.get("code_departement")),
            "type_marche": _as_scalar_str(ao.get("type_marche")),
            "descripteur_code": _as_scalar_str(ao.get("descripteur_code")),
            "descripteur_libelle": _as_scalar_str(ao.get("descripteur_libelle")),
            "url": ao.get("url"),
        }
        if emb is not None:
            row["embedding"] = emb
        rows.append(row)

    # Défense en profondeur : dédup par idweb dans le batch avant upsert.
    # Postgres refuse ON CONFLICT si 2 lignes de même clé dans le même batch.
    seen = {}
    for r in rows:
        seen[r["idweb"]] = r  # last-write-wins
    rows = list(seen.values())

    n_written = 0
    for i in range(0, len(rows), batch_size):        
        chunk = rows[i:i + batch_size]
        _client.table("boamp_ao").upsert(chunk, on_conflict="idweb").execute()
        n_written += len(chunk)
        log.info("  Supabase upsert AO : %d/%d", n_written, len(rows))
    return n_written


def _as_scalar_str(v) -> str | None:
    """Certains champs BOAMP arrivent en list — on aplatit en string."""
    if v is None:
        return None
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else None
    return str(v)


def get_missing_embedding_ids(idwebs: list[str]) -> list[str]:
    """Retourne les idwebs qui n'ont PAS encore d'embedding en DB."""
    if not idwebs:
        return []
    resp = (_client.table("boamp_ao")
            .select("idweb")
            .in_("idweb", idwebs)
            .is_("embedding", "null")
            .execute())
    return [r["idweb"] for r in resp.data]


def get_all_ao_ids_in_window(dateparution_min: str) -> set[str]:
    """Retourne tous les idwebs déjà stockés depuis dateparution_min (YYYY-MM-DD)."""
    resp = (_client.table("boamp_ao")
            .select("idweb")
            .gte("dateparution", dateparution_min)
            .execute())
    return {r["idweb"] for r in resp.data}


# ---------- PROFILE ----------

def upsert_profile(name: str, description: str,
                   embedding: np.ndarray | None = None) -> None:
    row = {"name": name, "description": description}
    if embedding is not None:
        row["embedding"] = _vec_to_pg(embedding)
    _client.table("profile").upsert(row, on_conflict="name").execute()


# ---------- MATCH ----------

def upsert_matches(profile_name: str, matches: list[dict],
                   llm_model: str = "gpt-4o-mini") -> int:
    """
    matches : liste de dicts avec ao_idweb, embedding_sim, llm_score, llm_reason, is_pepite
    """
    rows = [{
        "profile_name": profile_name,
        "ao_idweb": m["ao_idweb"],
        "embedding_sim": float(m["embedding_sim"]),
        "llm_score": int(m["llm_score"]),
        "llm_reason": m.get("llm_reason", ""),
        "llm_model": llm_model,
        "is_pepite": bool(m["is_pepite"]),
    } for m in matches]

    for i in range(0, len(rows), 100):
        _client.table("match") \
            .upsert(rows[i:i + 100], on_conflict="profile_name,ao_idweb") \
            .execute()
    return len(rows)


def get_pepites(profile_name: str, dateparution_min: str,
                min_score: int = 70) -> list[dict]:
    """
    Charge les pépites d'un profil sur une fenêtre temporelle,
    jointes aux données AO. Utilisé par le futur endpoint API pour le frontend.
    """
    resp = (_client.table("match")
            .select("llm_score, llm_reason, embedding_sim, "
                    "boamp_ao(idweb, objet, description, acheteur, "
                    "code_departement, dateparution, datelimitereponse, url)")
            .eq("profile_name", profile_name)
            .eq("is_pepite", True)
            .gte("llm_score", min_score)
            .gte("boamp_ao.dateparution", dateparution_min)
            .order("llm_score", desc=True)
            .execute())
    return resp.data

def match_ao_by_embedding(profile_embedding: np.ndarray,
                          date_min: str,
                          top_k: int = 30) -> list[dict]:
    """
    Appelle la fonction SQL pgvector côté Supabase.
    Retourne les top_k AO les plus proches sémantiquement du profil.
    """
    resp = _client.rpc("match_ao_by_embedding", {
        "query_embedding": _vec_to_pg(profile_embedding),
        "date_min": date_min,
        "top_k": top_k,
    }).execute()
    return resp.data

def get_last_successful_ingestion() -> dict | None:
    """Retourne le dernier ingestion réussi, ou None si aucun."""
    resp = (_client.table("ingestion_log")
            .select("*")
            .eq("status", "success")
            .order("finished_at", desc=True)
            .limit(1)
            .execute())
    return resp.data[0] if resp.data else None


def start_ingestion_log(days_back: int) -> int:
    """Crée une ligne 'running' et retourne son id."""
    resp = (_client.table("ingestion_log")
            .insert({"days_back": days_back, "status": "running"})
            .execute())
    return resp.data[0]["id"]


def finish_ingestion_log(log_id: int, n_new: int, n_updated: int) -> None:
    _client.table("ingestion_log").update({
        "status": "success",
        "finished_at": "now()",
        "n_new_ao": n_new,
        "n_updated_ao": n_updated,
    }).eq("id", log_id).execute()


def fail_ingestion_log(log_id: int, error: str) -> None:
    _client.table("ingestion_log").update({
        "status": "failed",
        "finished_at": "now()",
        "error": error[:500],
    }).eq("id", log_id).execute()

def get_and_increment_daily_counter(max_per_day: int) -> tuple[bool, int]:
    """
    Incrémente le compteur global du jour et retourne (autorisé, valeur_après_incrémentation).
    """
    from datetime import date
    today = date.today().isoformat()
    # Fetch current
    resp = (_client.table("rate_limit_global")
            .select("n_matches")
            .eq("day", today)
            .execute())
    current = resp.data[0]["n_matches"] if resp.data else 0
    if current >= max_per_day:
        return False, current
    # Upsert incrémenté
    new_value = current + 1
    _client.table("rate_limit_global").upsert(
        {"day": today, "n_matches": new_value},
        on_conflict="day"
    ).execute()
    return True, new_value