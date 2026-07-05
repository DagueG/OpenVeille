# -*- coding: utf-8 -*-
"""Client BOAMP : fetch paginé + parsing du champ `donnees` (FNSimple.initial)."""
import json
import logging
import time
import requests
from datetime import datetime, timedelta

from .config import BOAMP_URL, MIN_DESC_LENGTH

log = logging.getLogger(__name__)


def fetch_recent_ao(days_back: int = 7,
                    min_desc_length: int = MIN_DESC_LENGTH,
                    max_records: int | None = None) -> list[dict]:
    """
    Récupère les AO récents avec pagination (ODS API v2.1, limit=100 par page).
    Retourne uniquement les AO dont la description extraite fait >= min_desc_length chars.
    """
    date_min = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    all_raw = []
    offset = 0
    page_size = 100
    total = None

    while True:
        params = {
            "limit": page_size,
            "offset": offset,
            "order_by": "dateparution DESC",
            "where": f"dateparution >= '{date_min}'",
        }
        r = requests.get(BOAMP_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])

        if total is None:
            total = data.get("total_count", 0)
            log.info("BOAMP total sur %d jours (depuis %s) : %d AO",
                     days_back, date_min, total)

        if not results:
            break

        all_raw.extend(results)
        log.info("  Page offset=%d : +%d (cumul=%d/%d)",
                 offset, len(results), len(all_raw), total)

        offset += page_size
        if offset >= total:
            break
        time.sleep(0.1)  # politesse envers l'API publique

    enriched = []
    for rec in all_raw:
        parsed = _parse_record(rec)
        if parsed and len(parsed["description"]) >= min_desc_length:
            enriched.append(parsed)
            if max_records and len(enriched) >= max_records:
                break

    log.info("BOAMP filtré : %d AO retenus (desc >= %d chars) sur %d bruts",
             len(enriched), min_desc_length, len(all_raw))
    return enriched

def _parse_record(rec: dict) -> dict | None:
    """Extrait un dict AO propre depuis un record BOAMP brut."""
    donnees_raw = rec.get("donnees")
    donnees = {}
    if isinstance(donnees_raw, str):
        try:
            donnees = json.loads(donnees_raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(donnees_raw, dict):
        donnees = donnees_raw

    desc = _extract_description(donnees)
    if not desc:
        return None

    return {
        "idweb": rec.get("idweb"),
        "dateparution": rec.get("dateparution"),
        "datelimitereponse": rec.get("datelimitereponse"),
        "objet": (rec.get("objet") or "").strip(),
        "description": desc.strip(),
        "acheteur": rec.get("nomacheteur"),
        "code_departement": rec.get("code_departement"),
        "type_marche": rec.get("type_marche"),
        "descripteur_code": rec.get("descripteur_code"),
        "descripteur_libelle": rec.get("descripteur_libelle"),
        "url": f"https://www.boamp.fr/pages/avis/?q=idweb:%22{rec.get('idweb')}%22",
    }


def _extract_description(donnees: dict) -> str:
    """
    Extrait le texte descriptif depuis FNSimple.initial.
    Cible : natureMarche (titre + description), procedure (critères,
    capacités), informComplementaire (détails libres).
    """
    initial = donnees.get("FNSimple", {}).get("initial", {})
    if not initial:
        return ""

    parts = []

    nm = initial.get("natureMarche") or {}
    for key in ("intitule", "description", "lieuExecution"):
        v = nm.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    proc = initial.get("procedure") or {}
    for key in ("capaciteTech", "capaciteEcoFin", "capaciteExercice",
                "criteresAttrib", "categorieAcheteur"):
        v = proc.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    info = initial.get("informComplementaire") or {}
    _walk_strings(info, parts, min_len=50)

    # Dédoublonnage naïf sur les 80 premiers chars
    seen = set()
    unique = []
    for p in parts:
        k = p[:80]
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return "\n".join(unique)


def _walk_strings(node, acc: list, min_len: int = 50):
    if isinstance(node, dict):
        for v in node.values():
            _walk_strings(v, acc, min_len)
    elif isinstance(node, list):
        for item in node:
            _walk_strings(item, acc, min_len)
    elif isinstance(node, str):
        s = node.strip()
        if len(s) >= min_len and not s.startswith("http"):
            acc.append(s)