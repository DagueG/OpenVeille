# -*- coding: utf-8 -*-
"""
OpenVeille - Backend FastAPI.

Endpoints :
- GET  /health                       → liveness probe
- GET  /profiles                     → profils démo (pour référence, non utilisé par le front)
- GET  /stats                        → stats globales des profils démo (facultatif)
- POST /match                        → matching à la volée : description libre → pépites
"""
import os
import sys
import time
import logging
from hashlib import sha256
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

from openveille.db import get_client, match_ao_by_embedding  # noqa: E402
from openveille.matcher import embed_texts, rerank_llm_from_db_candidates  # noqa: E402
from openveille.profiles import PROFILES  # noqa: E402
from openveille.config import DEFAULT_NOTIF_THRESHOLD  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

app = FastAPI(
    title="OpenVeille API",
    description="Veille sémantique des marchés publics — API du portfolio thecoloss.com",
    version="0.2.0",
)

ALLOWED_ORIGINS = [
    "https://thecoloss.com",
    "https://www.thecoloss.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------- Cache TTL simple ----------
class TTLCache:
    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl
        self.store: dict[str, tuple[float, int, object]] = {}

    def get(self, key: str):
        entry = self.store.get(key)
        if not entry:
            return None
        ts, ttl, value = entry
        if time.time() - ts > ttl:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key: str, value, ttl: int | None = None):
        self.store[key] = (time.time(), ttl or self.default_ttl, value)


CACHE = TTLCache(default_ttl=60)


# ---------- Rate limiter mémoire (par IP) ----------
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max = max_requests
        self.window = window_seconds
        self.log: dict[str, list[float]] = {}

    def check_and_add(self, key: str) -> tuple[bool, int]:
        """Retourne (autorisé, temps_avant_reset_seconds)."""
        now = time.time()
        recent = [t for t in self.log.get(key, []) if now - t < self.window]
        if len(recent) >= self.max:
            oldest = min(recent)
            return False, int(self.window - (now - oldest))
        recent.append(now)
        self.log[key] = recent
        return True, 0


MATCH_LIMITER = RateLimiter(max_requests=5, window_seconds=3600)

_supabase = get_client()


# ---------- Helpers ----------
def _date_min(days: int) -> str:
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


def _client_ip(req: Request) -> str:
    # Render met la vraie IP dans X-Forwarded-For
    fwd = req.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return req.client.host if req.client else "unknown"


# ---------- Schémas Pydantic ----------
class MatchRequest(BaseModel):
    profile_description: str = Field(..., min_length=100, max_length=3000)
    days: int = Field(default=7)

    @field_validator("days")
    @classmethod
    def _validate_days(cls, v):
        if v not in (3, 7, 30):
            raise ValueError("days doit valoir 3, 7 ou 30")
        return v


# ---------- Endpoints simples ----------
@app.get("/health")
def health():
    return {"status": "ok", "at": datetime.utcnow().isoformat() + "Z"}


@app.get("/profiles")
def get_profiles():
    """Profils démo hardcodés (utile pour le sélecteur d'exemples côté front)."""
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
    """Chiffres du dataset BOAMP (nb AO en base sur la fenêtre) — pour la home."""
    key = f"stats:{days}"
    cached = CACHE.get(key)
    if cached is not None:
        return cached

    date_min = _date_min(days)
    ao_resp = (_supabase.table("boamp_ao")
               .select("idweb", count="exact")
               .gte("dateparution", date_min)
               .execute())
    data = {
        "window_days": days,
        "n_ao_total": ao_resp.count or 0,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
    CACHE.set(key, data, ttl=300)
    return data


# ---------- Endpoint principal : matching à la volée ----------
@app.post("/match")
def match(payload: MatchRequest, request: Request):
    """
    Matching sémantique en direct :
      1. Embed la description libre
      2. Recherche pgvector top-20 sur la fenêtre
      3. LLM re-rank
      4. Filtre par seuil (par défaut 70)
    """
    ip = _client_ip(request)
    ok, retry_after = MATCH_LIMITER.check_and_add(ip)
    if not ok:
        raise HTTPException(
            status_code=429,
            detail=f"Limite atteinte (5 analyses/heure). Réessayez dans {retry_after // 60} min.",
        )

    # Cache : mêmes params exacts = même réponse pendant 5 min
    desc = payload.profile_description.strip()
    cache_key = "match:" + sha256(f"{desc}|{payload.days}".encode()).hexdigest()
    cached = CACHE.get(cache_key)
    if cached is not None:
        log.info("Cache hit (ip=%s, days=%d)", ip, payload.days)
        return cached

    t0 = time.time()
    log.info("Match start (ip=%s, days=%d, desc_len=%d)", ip, payload.days, len(desc))

    # 1. Embed profil
    prof_vec = embed_texts([desc])[0]

    # 2. Recherche pgvector
    date_min = _date_min(payload.days)
    candidates = match_ao_by_embedding(prof_vec, date_min, top_k=20)

    # 3. Comptage total AO dans la fenêtre (pour affichage)
    n_ao_resp = (_supabase.table("boamp_ao")
                 .select("idweb", count="exact")
                 .gte("dateparution", date_min)
                 .execute())
    n_ao_scanned = n_ao_resp.count or 0
    
    # 4. LLM re-rank en parallèle
    reranked = rerank_llm_from_db_candidates(desc, candidates, max_workers=8)

    # 5. Filtre pépites
    pepites_raw = [
        r for r in reranked
        if r["llm_score"] >= DEFAULT_NOTIF_THRESHOLD and not r.get("llm_error")
    ]

    pepites = [{
        "score": r["llm_score"],
        "reason": r["llm_reason"],
        "similarity": round(float(r["similarity"]), 3),
        "idweb": r["idweb"],
        "objet": r["objet"],
        "description": (r.get("description") or "")[:500],
        "acheteur": r.get("acheteur"),
        "code_departement": r.get("code_departement"),
        "dateparution": r.get("dateparution"),
        "datelimitereponse": r.get("datelimitereponse"),
        "descripteur_code": r.get("descripteur_code"),
        "descripteur_libelle": r.get("descripteur_libelle"),
        "url": r["url"],
    } for r in pepites_raw]

    response = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "window_days": payload.days,
        "n_ao_scanned": n_ao_scanned,
        "n_candidates_reranked": len(reranked),
        "n_pepites": len(pepites),
        "threshold": DEFAULT_NOTIF_THRESHOLD,
        "duration_seconds": round(time.time() - t0, 2),
        "pepites": pepites,
    }

    CACHE.set(cache_key, response, ttl=300)
    log.info("Match done (ip=%s, %d pépites, %.1fs)",
             ip, len(pepites), response["duration_seconds"])
    return response