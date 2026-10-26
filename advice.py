"""
advice.py — Rice variety cooking & consumption advisory engine.

Given a detected rice variety (and simple quality signals already produced
by the detection/segmentation pipeline — grain counts and mask/box areas),
this module generates practical, general-purpose cooking and consumption
guidance. It adds no new ML models: everything here is deterministic,
rule-based logic layered on top of what YOLO + the classifier already output.

This is general culinary/informational guidance only — not personalized
dietary or medical advice. See the disclaimer appended to every result.
"""

import statistics

# --- Per-variety culinary reference data -----------------------------------
# water_ratio is rice:water by volume (absorption method) unless noted.
# Times are typical stovetop figures; rice cookers usually run a bit longer.
RICE_INFO = {
    "Arborio": {
        "origin": "Italy",
        "grain_type": "Short-grain, high-amylopectin (very starchy)",
        "texture": "Creamy, chewy, holds a slight bite (al dente) when done right",
        "water_ratio": "1 : 3 (added gradually, risotto method)",
        "soak_minutes": 0,
        "cook_minutes": "18-20",
        "best_dishes": ["Risotto", "Rice pudding", "Arancini"],
        "glycemic_index": "Medium-High",
        "storage_tip": "Airtight container, cool and dry; best used within 12-18 months.",
    },
    "Basmati": {
        "origin": "Indian subcontinent (Punjab region)",
        "grain_type": "Long-grain, aromatic, low stickiness",
        "texture": "Light and fluffy, grains stay separate after cooking",
        "water_ratio": "1 : 1.5",
        "soak_minutes": 20,
        "cook_minutes": "12-15",
        "best_dishes": ["Biryani", "Pulao / pilaf", "Plain steamed rice", "Jeera rice"],
        "glycemic_index": "Medium (lower when aged)",
        "storage_tip": "Cool, dry, airtight container; aged basmati (1+ yr) cooks fluffier.",
    },
    "Ipsala": {
        "origin": "Turkey (Ipsala / Edirne region)",
        "grain_type": "Medium-grain, Baldo-type",
        "texture": "Soft, slightly sticky, plump grains",
        "water_ratio": "1 : 1.75",
        "soak_minutes": 15,
        "cook_minutes": "15-18",
        "best_dishes": ["Turkish pilaf (pilav)", "Stuffed vegetables (dolma)", "Rice soup"],
        "glycemic_index": "Medium-High",
        "storage_tip": "Airtight container in a cool pantry; best texture within a year.",
    },
    "Jasmine": {
        "origin": "Thailand",
        "grain_type": "Long-grain, aromatic",
        "texture": "Soft, slightly sticky, fragrant aroma",
        "water_ratio": "1 : 1.25",
        "soak_minutes": 0,
        "cook_minutes": "15-18",
        "best_dishes": ["Thai curries", "Stir-fried rice", "Mango sticky rice", "Steamed rice"],
        "glycemic_index": "High",
        "storage_tip": "Airtight container; rinse before cooking to remove excess surface starch.",
    },
    "Karacadag": {
        "origin": "Turkey (Diyarbakır / Karacadağ region)",
        "grain_type": "Medium-grain heirloom",
        "texture": "Earthy flavor, plump and slightly chewy",
        "water_ratio": "1 : 1.75",
        "soak_minutes": 15,
        "cook_minutes": "16-18",
        "best_dishes": ["Traditional Turkish pilaf", "Stuffed grape leaves", "Soups"],
        "glycemic_index": "Medium",
        "storage_tip": "Store airtight, away from light and humidity.",
    },
    "Jhili": {
        "origin": "Bangladesh / Eastern India",
        "grain_type": "Fine, slender aromatic long-grain",
        "texture": "Light and fluffy, mild aroma",
        "water_ratio": "1 : 1.5",
        "soak_minutes": 15,
        "cook_minutes": "12-15",
        "best_dishes": ["Polao / pulao", "Biryani", "Everyday steamed rice"],
        "glycemic_index": "Medium",
        "storage_tip": "Airtight container in a cool, dry place.",
    },
    "HMT (Sona Masuri)": {
        "origin": "Andhra Pradesh / Karnataka, India",
        "grain_type": "Small/medium-grain, lightweight, low-starch",
        "texture": "Light, less sticky, easy to digest",
        "water_ratio": "1 : 2",
        "soak_minutes": 10,
        "cook_minutes": "12-15",
        "best_dishes": ["Everyday meals", "Curd rice", "Pongal", "Dosa/idli batter (parboiled)"],
        "glycemic_index": "Medium",
        "storage_tip": "Airtight container; this variety absorbs moisture readily, so keep it dry.",
    },
    "Masuri": {
        "origin": "South India",
        "grain_type": "Medium-grain, lightweight",
        "texture": "Soft, slightly sticky, mild flavor",
        "water_ratio": "1 : 2",
        "soak_minutes": 10,
        "cook_minutes": "12-15",
        "best_dishes": ["Daily meals", "Curd rice", "Khichdi", "Congee"],
        "glycemic_index": "Medium-High",
        "storage_tip": "Airtight container, cool and dry storage.",
    },
}

# Fallback for an unrecognized label (e.g. classifier missing, "Detected" placeholder).
_DEFAULT_INFO = {
    "origin": "Unknown",
    "grain_type": "Unclassified",
    "texture": "N/A",
    "water_ratio": "1 : 2 (general default)",
    "soak_minutes": 15,
    "cook_minutes": "15-18",
    "best_dishes": ["General steamed rice"],
    "glycemic_index": "Unknown",
    "storage_tip": "Store in an airtight container in a cool, dry place.",
}


def _polygon_area(mask_poly):
    """Shoelace-formula area for an (N,2) numpy polygon. Returns None if unusable."""
    if mask_poly is None or len(mask_poly) < 3:
        return None
    x = mask_poly[:, 0]
    y = mask_poly[:, 1]
    n = len(x)
    return float(0.5 * abs(sum(x[i] * y[i - 1] - x[i - 1] * y[i] for i in range(n))))


def assess_quality(majority_variety, grain_records, all_boxes, all_masks):
    """
    Heuristic quality/purity read on the batch, using only what the
    detection + segmentation pipeline already produced — no extra models.

    grain_records: list of (index_into_all_boxes/all_masks, variety) for
                    every grain that survived NMS + classification.

    Returns a dict with purity_pct, broken_pct, quality_tier,
    majority_count, total_count.
    """
    total_count = len(grain_records)
    majority_indices = [idx for idx, variety in grain_records if variety == majority_variety]
    majority_count = len(majority_indices)

    if total_count == 0 or majority_count == 0:
        return {
            "purity_pct": 0.0,
            "broken_pct": 0.0,
            "quality_tier": "No data",
            "majority_count": 0,
            "total_count": total_count,
        }

    areas = []
    for idx in majority_indices:
        area = _polygon_area(all_masks[idx])
        if area is None:
            x1, y1, x2, y2 = all_boxes[idx]
            area = max(0.0, (x2 - x1) * (y2 - y1))
        areas.append(area)

    median_area = statistics.median(areas) if areas else 0
    # Grains noticeably smaller than the batch's typical grain are treated
    # as broken/fragmented pieces rather than whole grains.
    broken_count = sum(1 for a in areas if median_area > 0 and a < 0.55 * median_area)

    purity_pct = 100.0 * majority_count / total_count
    broken_pct = 100.0 * broken_count / majority_count

    if broken_pct < 5:
        quality_tier = "Premium — mostly whole, uniform grains"
    elif broken_pct < 15:
        quality_tier = "Good — minor breakage present"
    else:
        quality_tier = "Mixed/Broken — noticeable fragment content"

    return {
        "purity_pct": purity_pct,
        "broken_pct": broken_pct,
        "quality_tier": quality_tier,
        "majority_count": majority_count,
        "total_count": total_count,
    }


def generate_consumption_advice(majority_variety, quality_info):
    """
    Build a short, general culinary advisory string for the detected batch.
    Purely informational/culinary guidance — not personalized dietary or
    medical advice (see disclaimer in the output).
    """
    if majority_variety is None or quality_info["total_count"] == 0:
        return "No grains were detected, so no consumption advice is available."

    info = RICE_INFO.get(majority_variety, _DEFAULT_INFO)
    broken_pct = quality_info["broken_pct"]
    purity_pct = quality_info["purity_pct"]

    best_dishes = list(info["best_dishes"])
    quality_note = ""
    if broken_pct >= 15:
        # Broken/fragmented rice doesn't hold its shape in dishes that need
        # long, intact grains (biryani, pulao) — steer toward dishes that
        # use broken rice well instead.
        best_dishes = ["Congee / porridge", "Idli / dosa batter", "Kheer / rice pudding", "Rice flour"]
        quality_note = (
            f"\n\n> Note: about {broken_pct:.0f}% of the {majority_variety} grains look fragmented, "
            "so dishes that need long intact grains (biryani, pulao) may disappoint — "
            "the alternatives above suit broken rice better."
        )

    lines = [
        f"**Detected variety:** {majority_variety}  "
        f"({purity_pct:.0f}% of {quality_info['total_count']} grains detected)",
        f"**Batch quality:** {quality_info['quality_tier']} (~{broken_pct:.0f}% fragmented)",
        f"**Origin:** {info['origin']}  |  **Grain type:** {info['grain_type']}",
        f"**Cooking guide:** {info['water_ratio']} water, "
        f"soak {info['soak_minutes']} min, cook {info['cook_minutes']} min",
        f"**Best used for:** {', '.join(best_dishes)}{quality_note}",
        f"**Typical glycemic index:** {info['glycemic_index']} "
        "(general reference — actual GI varies with cooking method, aging, and serving size)",
        f"**Storage tip:** {info['storage_tip']}",
        "",
        "_General culinary guidance only, not personalized dietary or medical advice. "
        "For diabetes management or other health conditions, consult a doctor or registered dietitian._",
    ]
    return "\n\n".join(lines)
