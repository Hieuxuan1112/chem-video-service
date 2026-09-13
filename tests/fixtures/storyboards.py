"""Schema-valid storyboard payloads for gate tests (content quality is not the point here)."""

from typing import Any


def minimal_storyboard_dict() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "topic_id": "ph_scale",
        "title": "How the pH scale works",
        "learning_objective": "Explain what pH numbers mean for acids and bases.",
        "scenes": [
            {
                "on_screen_title": "Your question",
                "narration": "Let's explore the pH scale.",
                "visual": {"type": "title_card", "subtitle": "Acids, bases and neutral"},
            },
            {
                "on_screen_title": "The scale",
                "narration": "pH runs from 0 to 14.",
                "visual": {
                    "type": "ph_scale",
                    "markers": ["lemon_juice", "bleach"],
                    "highlight": "none",
                },
            },
            {
                "on_screen_title": "Ten times",
                "narration": "Each step is ten times.",
                "visual": {"type": "log_ratio", "from_ph": 3, "to_ph": 4},
            },
            {
                "on_screen_title": "Compare",
                "narration": "Acids versus bases.",
                "visual": {
                    "type": "comparison_table",
                    "left_header": "Acids",
                    "right_header": "Bases",
                    "rows": [
                        {"aspect": "pH", "left": "Below 7", "right": "Above 7"},
                        {"aspect": "Example", "left": "Lemon juice", "right": "Bleach"},
                    ],
                },
            },
            {
                "on_screen_title": "Key points",
                "narration": "Remember these ideas.",
                "visual": {
                    "type": "key_points",
                    "points": ["Below 7 is acidic", "Above 7 is basic"],
                },
            },
        ],
    }
