# services/schema_detect.py
# ============================
from typing import Dict, Optional, List
import re

CANON = {
    "description": [
        "ticket description", "description", "issue description", "problem description",
        "message", "body", "details"
    ],
    "priority": [
        "ticket priority", "priority", "urgency", "severity"
    ],
    "satisfaction": [
        "customer satisfaction rating", "csat", "satisfaction", "rating", "customer rating"
    ],
    "channel": [
        "ticket channel", "channel", "contact channel", "source", "communication channel"
    ],
    "status": [
        "ticket status", "status", "state"
    ],
    "id": [
        "ticket id", "id", "case id", "request id"
    ]
}

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())

def _find_best(columns: List[str], synonyms: List[str]) -> Optional[str]:
    norm_cols = [_norm(c) for c in columns]
    syns = [_norm(s) for s in synonyms]
    # exact
    for i, c in enumerate(norm_cols):
        if c in syns:
            return columns[i]
    # partial
    for i, c in enumerate(norm_cols):
        if any(s in c or c in s for s in syns):
            return columns[i]
    return None

def detect_schema(columns: List[str]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for key, syns in CANON.items():
        mapping[key] = _find_best(columns, syns)
    return mapping
