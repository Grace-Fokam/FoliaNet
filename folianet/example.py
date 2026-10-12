"""Offline response fixture. No image, trained model, or provider is evaluated."""

import json

from folianet.diseases import get_disease
from folianet.recommend import recommend_action


def example_response() -> dict:
    """Return a fresh, explicitly illustrative prediction-shaped response."""
    disease = get_disease("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    risk = 0.7  # Hand-authored illustration, not a computed prediction.
    band, action = recommend_action(disease.key, False, risk)
    return {
        "mode": "illustrative_example",
        "is_prediction": False,
        "notice": "Hand-authored example only. No image analyzed, checkpoint loaded, "
                  "or weather/satellite data retrieved. Do not use for field decisions.",
        "risk_score": risk,
        "risk_score_pct": 70.0,
        "risk_band": band,
        "confidence": {"score": 0.75, "level": "high"},
        "likely_stressor": {
            "name": disease.display_name, "pathogen": disease.pathogen,
            "crop": disease.crop, "healthy": False,
        },
        "recommended_action": action,
        "details": {
            "image_disease_probability": 0.8,
            "environmental_favorability": 0.55,
            "class_probabilities": {
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 0.7,
                "Corn_(maize)___Common_rust_": 0.05,
                "Corn_(maize)___Northern_Leaf_Blight": 0.05,
                "Corn_(maize)___healthy": 0.2,
            },
            "weather": {
                "temp_mean_c": 26.0, "humidity_mean_pct": 90.0,
                "precip_7d_mm": 20.0, "leaf_wetness_hours": 12.0,
                "gdd": 112.0, "data_available": False,
            },
            "ndvi": 0.62,
            "data_sources": {"weather": False, "satellite": False},
            "fusion_weights": {"image": 0.6, "environment": 0.4},
            "model_version": None,
            "provenance": {
                "image": "hand_authored_example_no_image_analyzed",
                "weather": "hand_authored_example_not_retrieved",
                "ndvi": "hand_authored_example_not_retrieved",
                "scores": "hand_authored_example_not_model_outputs",
                "recommendation": "existing_rule_based_advisory_text",
                "fusion_weights": "illustrative_only_not_executed",
            },
        },
    }


if __name__ == "__main__":
    print(json.dumps(example_response(), indent=2, allow_nan=False))
