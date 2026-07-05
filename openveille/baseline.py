# -*- coding: utf-8 -*-
"""
Baseline "veille classique" façon Wanao/Doubletrade :
filtre par mots-clés (dans objet + description) et par famille de descripteurs BOAMP.

Objectif : reproduire ce qu'une PME obtient avec une config de veille standard,
pour objectiver le gain apporté par le matching sémantique.
"""
import logging
import re
import unicodedata

log = logging.getLogger(__name__)


# Configurations baseline par profil.
# Ces mots-clés / familles sont ce qu'une PME configurerait raisonnablement
# dans un outil de veille classique en pensant "voici mon métier".
#
# On garde ça volontairement PLAUSIBLE (pas trop généreux, pas trop restrictif) —
# c'est ce qui rend la comparaison honnête.
BASELINE_CONFIG = {
    "IT_data_platform": {
        "keywords": [
            "informatique", "logiciel", "application", "développement",
            "data", "donnée", "base de données", "sql",
            "power bi", "tableau de bord", "décisionnel", "business intelligence",
            "esn", "prestation intellectuelle",
            "python", "typescript", "api",
            "cloud", "hébergement",
            "portail", "site web", "plateforme numérique",
        ],
        # Familles BOAMP typiques pour prestations IT
        "descripteur_families": ["163", "165", "166", "155"],
    },
    "BTP_renovation_energetique": {
        "keywords": [
            "rénovation", "réhabilitation", "restructuration",
            "isolation", "ite", "étanchéité",
            "menuiserie", "fenêtre", "toiture", "couverture",
            "chauffage", "pompe à chaleur", "pac ", "vmc", "ventilation",
            "cvc", "climatisation",
            "thermique", "énergétique", "performance énergétique", "cee",
            "bâtiment", "école", "gymnase", "ehpad", "logement social",
            "maîtrise d'œuvre bâtiment", "moe bâtiment",
        ],
        "descripteur_families": ["33", "234", "365", "197", "105", "253"],
    },
    "Consulting_transformation_publique": {
        "keywords": [
            "conseil", "consulting", "cabinet",
            "accompagnement", "assistance à maîtrise d'ouvrage", "amoa", "amo",
            "schéma directeur", "audit organisationnel",
            "conduite du changement", "transformation",
            "évaluation politique publique", "évaluation de politique",
            "modernisation", "processus métier",
            "formation management", "coaching",
            "prestation intellectuelle", "études",
        ],
        "descripteur_families": ["274", "155", "163"],
    },
}


def _normalize(text: str) -> str:
    """Casse basse + suppression accents pour matcher sans sensibilité."""
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def _match_keyword(hay: str, needle: str) -> bool:
    """Match mot-clé avec limites de mots pour éviter les faux positifs."""
    hay_n = _normalize(hay)
    needle_n = _normalize(needle)
    # \b ne marche pas bien avec des tokens contenant espaces/apostrophes,
    # donc on cherche entouré de bornes non-alphanumériques ou début/fin.
    pattern = r"(?:^|[^a-z0-9])" + re.escape(needle_n) + r"(?:[^a-z0-9]|$)"
    return bool(re.search(pattern, hay_n))


def apply_baseline(profile_name: str, aos: list[dict]) -> list[dict]:
    """
    Applique la baseline (mots-clés + descripteurs) sur une liste d'AO.
    Retourne les AO matchés, enrichis de :
      - baseline_matched (bool)
      - baseline_matched_keywords (list[str])
      - baseline_matched_family (str ou "")
    """
    cfg = BASELINE_CONFIG.get(profile_name)
    if not cfg:
        raise ValueError(f"Pas de config baseline pour profil : {profile_name}")

    keywords = cfg["keywords"]
    families = set(cfg["descripteur_families"])

    matched = []
    for ao in aos:
        hay = f"{ao.get('objet', '')} {ao.get('description', '')}"
        hit_kw = [k for k in keywords if _match_keyword(hay, k)]

        desc_code = str(ao.get("descripteur_code") or "")
        hit_family = ""
        for fam in families:
            if desc_code.startswith(fam):
                hit_family = fam
                break

        if hit_kw or hit_family:
            matched.append({
                **ao,
                "baseline_matched": True,
                "baseline_matched_keywords": hit_kw,
                "baseline_matched_family": hit_family,
            })

    log.info("Baseline %s : %d/%d AO matchés (mots-clés OU descripteur)",
             profile_name, len(matched), len(aos))
    return matched


def compare_sets(pipeline_pepites: list[dict],
                 baseline_matches: list[dict]) -> dict:
    """
    Compare l'ensemble des pépites (sémantique + LLM) vs les matches baseline.
    Retourne un dict avec :
      - both : AO trouvés par les deux → confirmation
      - only_semantic : AO trouvés uniquement par sémantique → LA killer feature
      - only_baseline : AO trouvés uniquement par baseline → potentiels FP baseline
      - stats
    """
    pep_ids = {a["idweb"] for a in pipeline_pepites}
    bas_ids = {a["idweb"] for a in baseline_matches}

    pep_by_id = {a["idweb"]: a for a in pipeline_pepites}
    bas_by_id = {a["idweb"]: a for a in baseline_matches}

    both_ids = pep_ids & bas_ids
    only_sem_ids = pep_ids - bas_ids
    only_bas_ids = bas_ids - pep_ids

    return {
        "both": [pep_by_id[i] for i in both_ids],
        "only_semantic": [pep_by_id[i] for i in only_sem_ids],
        "only_baseline": [bas_by_id[i] for i in only_bas_ids],
        "stats": {
            "n_pipeline_pepites": len(pep_ids),
            "n_baseline_matches": len(bas_ids),
            "n_intersection": len(both_ids),
            "n_only_semantic": len(only_sem_ids),
            "n_only_baseline": len(only_bas_ids),
        },
    }