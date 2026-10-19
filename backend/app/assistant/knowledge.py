"""Small local knowledge base with keyword retrieval.

The notes below are project-authored summaries of widely known O&M practice, written for
this demo. They are not a substitute for manufacturer manuals or standards such as
IEC 62446-1; the assistant always shows them as *retrieved guidance*, separate from
anything the model generates.
"""
from app.schemas import RetrievedSource

SOURCE_LABEL = "SolarSight guidance notes (project-authored summary)"

GUIDANCE_NOTES = [
    {
        "id": "om-soiling",
        "title": "Soiling and dust",
        "keywords": ["dust", "dirty", "soiling", "clean", "dusty"],
        "text": "Dust and soiling reduce irradiance reaching the cells. Compare output with a clean reference "
        "string before and after cleaning, and follow the module maker's cleaning guidance (water quality, tools).",
    },
    {
        "id": "om-bird-drop",
        "title": "Bird droppings and localized shading",
        "keywords": ["bird", "drop", "droppings", "shading", "hotspot"],
        "text": "Localized shading can cause hot spots and mismatch losses. Inspect the affected module, "
        "clean it per manufacturer guidance, and check for discoloration or cell damage afterwards.",
    },
    {
        "id": "om-snow",
        "title": "Snow cover",
        "keywords": ["snow", "frost", "ice"],
        "text": "Snow blocks irradiance almost completely. Do not use sharp tools or force on modules; "
        "wait for melt or use manufacturer-approved methods.",
    },
    {
        "id": "om-physical",
        "title": "Cracks and physical damage",
        "keywords": ["crack", "physical", "damage", "broken", "glass", "delamination"],
        "text": "Cracked glass or cells can allow moisture ingress and electrical faults. Treat damaged modules as "
        "a safety concern, restrict access, and arrange assessment by qualified personnel.",
    },
    {
        "id": "om-electrical",
        "title": "Electrical damage and hot spots",
        "keywords": ["electrical", "burn", "hot", "fault", "thermal", "junction"],
        "text": "Burn marks or suspected electrical faults need a qualified electrician. Thermal (infrared) "
        "imaging under load and I–V curve testing are common follow-ups. De-energize only under proper procedure.",
    },
    {
        "id": "om-underperformance",
        "title": "Investigating lower-than-forecast output",
        "keywords": ["underperformance", "forecast", "lower", "output", "energy", "deviation", "predicted"],
        "text": "Check data quality first (meters, timestamps), then weather inputs versus actual conditions, "
        "then inverter alarms and string currents, then physical inspection. Forecast error is expected: "
        "compare the deviation with the model's reported test error before suspecting a fault.",
    },
    {
        "id": "om-cloud-uncertainty",
        "title": "Forecast uncertainty and cloud cover",
        "keywords": ["cloud", "uncertainty", "interval", "cloudy", "weather", "synthetic"],
        "text": "Cloud cover is the largest source of irradiance forecast error. Treat single-day scenarios "
        "as indicative, and prefer the lower bound for commitments that carry financial risk.",
    },
    {
        "id": "om-safety",
        "title": "Safety before maintenance",
        "keywords": ["safety", "maintenance", "action", "review", "technician"],
        "text": "PV arrays stay energized in daylight. Any maintenance decision must be reviewed and carried out "
        "by qualified staff following site safety procedures and lock-out/tag-out where applicable.",
    },
]


def retrieve_guidance(
    search_terms: list[str], priority_terms: list[str] | None = None, limit: int = 3
) -> list[RetrievedSource]:
    """Rank notes by keyword hits. A hit in priority_terms (e.g. the predicted panel
    condition) counts triple, so a specific finding outranks generic forecast notes."""
    haystack = " ".join(term.lower() for term in search_terms)
    priority_haystack = " ".join(term.lower() for term in (priority_terms or []))
    scored = []
    for note in GUIDANCE_NOTES:
        score = sum(1 for keyword in note["keywords"] if keyword in haystack)
        score += 3 * sum(1 for keyword in note["keywords"] if keyword in priority_haystack)
        if score:
            scored.append((score, note))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected = [note for _, note in scored[:limit]]
    safety_note = next(note for note in GUIDANCE_NOTES if note["id"] == "om-safety")
    if safety_note not in selected:
        selected = selected[: limit - 1] + [safety_note]  # always surface the safety note
    return [
        RetrievedSource(id=n["id"], title=f"{n['title']} — {SOURCE_LABEL}", excerpt=n["text"]) for n in selected
    ]
