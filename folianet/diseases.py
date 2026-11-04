"""Corn and wheat diseases and the weather each one likes.

Keys match the training image folder names. The weather ranges are rough
guesses, so test them on real field data. Add new diseases here."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Favorability:
    """Temperature, humidity, and wetness ranges for one disease."""
    t_min: float          # too cold below this (C)
    t_opt_low: float      # best temp range starts here
    t_opt_high: float     # best temp range ends here
    t_max: float          # too hot above this
    humidity_min: float   # too dry below this humidity (%)
    wetness_hours: float  # wet hours needed for max wetness score


_DEFAULT_FAVORABILITY = Favorability(15, 22, 28, 35, 75, 12)


@dataclass(frozen=True)
class Disease:
    key: str              # short id
    crop: str             # "Corn" or "Wheat"
    display_name: str     # readable name
    pathogen: str         # what causes it
    is_healthy: bool = False
    favorability: Favorability = field(default_factory=lambda: _DEFAULT_FAVORABILITY)


# Keyed by image folder name.
DISEASES: Dict[str, Disease] = {
    # ---- Corn / maize ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": Disease(
        key="corn_gray_leaf_spot",
        crop="Corn",
        display_name="Gray leaf spot",
        pathogen="Cercospora zeae-maydis",
        favorability=Favorability(18, 24, 30, 35, 85, 12),
    ),
    "Corn_(maize)___Common_rust_": Disease(
        key="corn_common_rust",
        crop="Corn",
        display_name="Common rust",
        pathogen="Puccinia sorghi",
        favorability=Favorability(10, 16, 23, 30, 80, 6),
    ),
    "Corn_(maize)___Northern_Leaf_Blight": Disease(
        key="corn_northern_leaf_blight",
        crop="Corn",
        display_name="Northern leaf blight",
        pathogen="Exserohilum turcicum",
        favorability=Favorability(15, 18, 27, 32, 85, 6),
    ),
    "Corn_(maize)___healthy": Disease(
        key="corn_healthy",
        crop="Corn",
        display_name="Healthy",
        pathogen="—",
        is_healthy=True,
        favorability=Favorability(0, 0, 0, 0, 0, 0),
    ),

    # ---- Wheat ----
    "Wheat___Leaf_rust": Disease(
        key="wheat_leaf_rust",
        crop="Wheat",
        display_name="Leaf (brown) rust",
        pathogen="Puccinia triticina",
        favorability=Favorability(10, 15, 22, 30, 80, 6),
    ),
    "Wheat___Stripe_rust": Disease(
        key="wheat_stripe_rust",
        crop="Wheat",
        display_name="Stripe (yellow) rust",
        pathogen="Puccinia striiformis",
        favorability=Favorability(5, 10, 18, 25, 85, 6),
    ),
    "Wheat___Septoria": Disease(
        key="wheat_septoria",
        crop="Wheat",
        display_name="Septoria leaf blotch",
        pathogen="Zymoseptoria tritici",
        favorability=Favorability(10, 16, 22, 28, 90, 24),
    ),
    "Wheat___healthy": Disease(
        key="wheat_healthy",
        crop="Wheat",
        display_name="Healthy",
        pathogen="—",
        is_healthy=True,
        favorability=Favorability(0, 0, 0, 0, 0, 0),
    ),
}


def get_disease(class_name: str) -> Disease:
    """Look up a disease, or make a basic entry if it isn't listed."""
    if class_name in DISEASES:
        return DISEASES[class_name]
    # Unknown class: guess the crop from the name and use default rules.
    crop = "Wheat" if "wheat" in class_name.lower() else "Corn"
    healthy = "healthy" in class_name.lower()
    return Disease(
        key=class_name.lower().replace(" ", "_"),
        crop=crop,
        display_name=class_name.split("___")[-1].replace("_", " ").strip() or class_name,
        pathogen="unknown",
        is_healthy=healthy,
        favorability=Favorability(15, 22, 28, 35, 80, 12),
    )
