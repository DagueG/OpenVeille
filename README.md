# OpenVeille

OpenVeille est une application de veille sémantique des marchés publics BOAMP. Le projet ingère les avis, calcule des embeddings, applique un re-rank LLM, puis expose une API FastAPI pour rechercher des opportunités adaptées à un profil d'entreprise.

## Vue d'ensemble

Le pipeline repose sur trois briques principales :

- ingestion BOAMP vers Supabase,
- recherche vectorielle sur les avis stockés,
- re-rank par modèle de langage pour filtrer les vraies pépites.

Le dépôt contient à la fois les scripts de traitement batch et l'API utilisée pour les tests et pour le frontend.

## Arborescence utile

- `api/main.py` : point d'entrée FastAPI.
- `openveille/boamp_client.py` : récupération des avis BOAMP.
- `openveille/ingestion.py` : ingestion incrémentale et journaux d'ingestion.
- `openveille/db.py` : accès Supabase, upserts, recherche et compteurs.
- `openveille/matcher.py` : embeddings et re-rank LLM.
- `openveille/profiles.py` : profils d'entreprise de démonstration.
- `05_ingest_boamp.py` : script CLI d'ingestion.
- `03_run_pipeline.py` : pipeline complet BOAMP → embedding → re-rank.
- `04_compare_baseline.py` : comparaison avec le baseline.
- `01_test_embeddings_20pairs.py` : test d'embeddings.
- `02_llm_rerank.py` : test de re-rank LLM.
- `06_run_pipeline_db.py` : variante du pipeline basée sur la base de données.

## Prérequis

- Python 3.11 ou compatible avec l'environnement local.
- `uv` pour lancer les scripts.
- Une base Supabase avec les tables et RPC attendues par le projet.
- Des variables d'environnement configurées dans un fichier `.env` à la racine.

## Variables d'environnement

Le projet lit les variables suivantes :

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_EMBED_DEPLOYMENT`
- `AZURE_CHAT_DEPLOYMENT`

Un exemple est disponible dans `api/.env.example`.

## Installation

Installer les dépendances Python :

```powershell
uv sync
```

Si votre environnement n'utilise pas `uv sync`, installez au minimum les dépendances listées dans `api/requirements.txt`.

## Lancer l'API

```powershell
uv run uvicorn api.main:app --reload --port 8000
```

Endpoints principaux :

- `GET /health` : état du service.
- `GET /profiles` : profils de démonstration.
- `GET /stats?days=7` : nombre d'avis sur la fenêtre demandée.
- `POST /match` : matching d'un profil texte contre les avis récents.

Exemple de requête PowerShell :

```powershell
$body = @{
    profile_description = @'
Nous sommes une entreprise générale du bâtiment de 80 salariés implantée en région Auvergne-Rhône-Alpes, spécialisée dans la rénovation énergétique de bâtiments publics. Nos métiers : isolation thermique par l'extérieur, menuiseries, chauffage et ventilation, étanchéité. Nous réalisons des marchés de rénovation d'écoles, gymnases, EHPAD, bureaux administratifs. Nos chantiers vont de 200 k€ à 8 M€. Nous avons les qualifications Qualibat RGE.
'@
    days = 7
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://localhost:8000/match" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
```

## Ingestion BOAMP

Le script d'ingestion charge les avis récents dans Supabase :

```powershell
uv run python 05_ingest_boamp.py
uv run python 05_ingest_boamp.py --days 30
uv run python 05_ingest_boamp.py --force-reembed
```

L'API peut aussi déclencher un refresh incrémental en arrière-plan si la dernière ingestion est trop ancienne.

## Pipeline complet

Pour exécuter le pipeline complet en local :

```powershell
uv run python 03_run_pipeline.py
uv run python 03_run_pipeline.py --days 3
uv run python 03_run_pipeline.py --profile BTP_renovation_energetique
```

Options utiles :

- `--top-k` : nombre d'avis envoyés au LLM.
- `--threshold` : score minimal pour considérer une pépite.
- `--max-ao` : limite de test pour raccourcir l'exécution.

## Données et stockage

Le projet s'appuie sur Supabase pour :

- stocker les avis BOAMP,
- stocker les profils,
- stocker les matchs et les journaux d'ingestion,
- appeler une RPC de recherche vectorielle.

Les fonctions attendues côté base sont notamment :

- `match_ao_by_embedding`
- tables `boamp_ao`, `profile`, `match`, `ingestion_log`, `rate_limit_global`

## Profils de démonstration

Le projet inclut plusieurs profils de test dans `openveille/profiles.py`, notamment :

- `IT_data_platform`
- `BTP_renovation_energetique`
- `Consulting_transformation_publique`

## Notes de fonctionnement

- L'API applique un rate limit mémoire par client sur `/match`.
- Les réponses `/match` sont mises en cache en mémoire pendant une courte durée.
- Les pépites sont déterminées à partir du score LLM défini dans la configuration.

## Développement

Les fichiers de scripts numérotés servent de jalons d'expérimentation. Pour comprendre une brique, le point d'entrée le plus utile est souvent :

- `openveille/ingestion.py` pour l'ingestion,
- `openveille/matcher.py` pour le scoring,
- `openveille/db.py` pour la persistance,
- `api/main.py` pour l'API.
