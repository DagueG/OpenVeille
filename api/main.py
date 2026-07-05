# -*- coding: utf-8 -*-
"""
OpenVeille - Backend FastAPI (read-only).

Sert le frontend thecoloss.com :
- GET /health                       → liveness probe
- GET /profiles                     → liste des profils démo disponibles
- GET /stats                        → stats globales pour la home (nb AO, pépites, etc.)
- GET /pepites/{profile_name}       → pépites d'un profil sur une fenêtre temporelle

Design :
- Lecture seule vers Supabase via service_role (pas de mutation ici)
- Cache mémoire simple avec TTL (les données changent peu, on évite de spammer Supabase)
- CORS ouvert sur thecoloss.com + localhost dev
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Le module openveille est un cran plus haut
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from openveille.db import get_client  # noqa: E402
from openveille.profiles import PROFILES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

app = FastAPI(
    title="OpenVeille API",
    description="Veille sémantique des marchés publics — API du portfolio thecoloss.com",
    version="0.1.0",
)

# CORS : autoriser le frontend + le dev local Vite (5173)
ALLOWED_ORIGINS = [
    "https://thecoloss.com",
    "https://www.thecoloss.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------- Cache mémoire ultra-simple ----------
class TTLCache:
    """Cache dict + TTL. Suffisant pour un backend mono-instance sur Railway/Render."""
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self.store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        entry = self.store.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key: str, value):
        self.store[key] = (time.time(), value)


CACHE = TTLCache(ttl_seconds=60)
_supabase = get_client()


# ---------- Helpers ----------
def _date_min(days: int) -> str:
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


def _fetch_stats(days: int) -> dict:
    """Stats globales pour la home : count AO, count pépites par profil."""
    date_min = _date_min(days)

    # Count AO dans la fenêtre
    ao_resp = (_supabase.table("boamp_ao")
               .select("idweb", count="exact")
               .gte("dateparution", date_min)
               .execute())
    n_ao = ao_resp.count or 0

    # Count pépites par profil (jointure pour respecter la fenêtre AO)
    profiles_stats = []
    for prof_name in PROFILES.keys():
        # On récupère les idweb pépites du profil, puis on filtre côté DB via la jointure implicite
        # via une requête sur match avec select imbriqué sur boamp_ao pour respecter la fenêtre.
        resp = (_supabase.table("match")
                .select("ao_idweb, llm_score, boamp_ao!inner(dateparution)")
                .eq("profile_name", prof_name)
                .eq("is_pepite", True)
                .gte("boamp_ao.dateparution", date_min)
                .execute())
        rows = resp.data or []
        profiles_stats.append({
            "name": prof_name,
            "n_pepites": len(rows),
        })

    return {
        "window_days": days,
        "n_ao_total": n_ao,
        "profiles": profiles_stats,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }


def _fetch_pepites(profile_name: str, days: int, min_score: int) -> list[dict]:
    """Retourne les pépites d'un profil avec leur AO joint, triées par score."""
    if profile_name not in PROFILES:
        raise HTTPException(status_code=404, detail=f"Profil inconnu : {profile_name}")

    date_min = _date_min(days)
    resp = (_supabase.table("match")
            .select("llm_score, llm_reason, embedding_sim, computed_at, "
                    "boamp_ao!inner(idweb, objet, description, acheteur, "
                    "code_departement, dateparution, datelimitereponse, "
                    "descripteur_code, descripteur_libelle, url)")
            .eq("profile_name", profile_name)
            .eq("is_pepite", True)
            .gte("llm_score", min_score)
            .gte("boamp_ao.dateparution", date_min)
            .order("llm_score", desc=True)
            .execute())

    out = []
    for row in resp.data or []:
        ao = row.get("boamp_ao") or {}
        out.append({
            "score": row["llm_score"],
            "reason": row["llm_reason"],
            "similarity": round(float(row["embedding_sim"]), 3),
            "idweb": ao.get("idweb"),
            "objet": ao.get("objet"),
            "description": (ao.get("description") or "")[:800],
            "acheteur": ao.get("acheteur"),
            "code_departement": ao.get("code_departement"),
            "dateparution": ao.get("dateparution"),
            "datelimitereponse": ao.get("datelimitereponse"),
            "descripteur_code": ao.get("descripteur_code"),
            "descripteur_libelle": ao.get("descripteur_libelle"),
            "url": ao.get("url"),
        })
    return out


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "at": datetime.utcnow().isoformat() + "Z"}


@app.get("/profiles")
def get_profiles():
    """Liste des profils démo avec leur description (pour la sidebar du dashboard)."""
    return [
        {
            "name": name,
            "description": desc.strip(),
            "label": name.replace("_", " "),
        }
        for name, desc in PROFILES.items()
    ]


@app.get("/stats")
def get_stats(days: int = Query(7, ge=1, le=90)):
    """Chiffres pour la home : nb AO scannés, pépites par profil."""
    key = f"stats:{days}"
    cached = CACHE.get(key)
    if cached is not None:
        return cached
    data = _fetch_stats(days)
    CACHE.set(key, data)
    return data


@app.get("/pepites/{profile_name}")
def get_pepites(profile_name: str,
                days: int = Query(7, ge=1, le=90),
                min_score: int = Query(70, ge=0, le=100)):
    """Pépites d'un profil sur les derniers `days` jours, score LLM >= min_score."""
    key = f"pepites:{profile_name}:{days}:{min_score}"
    cached = CACHE.get(key)
    if cached is not None:
        return cached
    data = _fetch_pepites(profile_name, days, min_score)
    CACHE.set(key, data)
    return data