# -*- coding: utf-8 -*-
"""
Logique d'ingestion BOAMP réutilisable par le script CLI et par l'API.
"""
import logging
from datetime import datetime, timedelta

from .boamp_client import fetch_recent_ao
from .matcher import embed_texts
from .db import (
    upsert_boamp_ao_batch,
    get_all_ao_ids_in_window,
    start_ingestion_log,
    finish_ingestion_log,
    fail_ingestion_log,
)

log = logging.getLogger(__name__)


def run_ingestion(days_back: int = 7, force_reembed: bool = False) -> dict:
    """
    Exécute une ingestion complète. Log le résultat dans ingestion_log.
    Retourne un dict avec le résumé.
    """
    date_min = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    log_id = start_ingestion_log(days_back)

    try:
        aos = fetch_recent_ao(days_back=days_back)
        if not aos:
            finish_ingestion_log(log_id, 0, 0)
            return {"n_new": 0, "n_updated": 0}

        existing_ids = get_all_ao_ids_in_window(date_min) if not force_reembed else set()
        new_or_updated = [a for a in aos if a["idweb"] not in existing_ids]
        already_present = [a for a in aos if a["idweb"] in existing_ids]

        if new_or_updated:
            texts = [f"{a['objet']}\n\n{a['description']}" for a in new_or_updated]
            vecs = embed_texts(texts)
            upsert_boamp_ao_batch(new_or_updated, embeddings=vecs)

        if already_present:
            upsert_boamp_ao_batch(already_present, embeddings=None)

        finish_ingestion_log(log_id, len(new_or_updated), len(already_present))
        log.info("Ingestion done: %d new, %d updated", len(new_or_updated), len(already_present))
        return {"n_new": len(new_or_updated), "n_updated": len(already_present)}

    except Exception as e:
        log.exception("Ingestion failed")
        fail_ingestion_log(log_id, f"{type(e).__name__}: {e}")
        raise