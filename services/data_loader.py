# services/data_loader.py
# ============================
import os
import pandas as pd
from typing import Optional, Tuple
from services.schema_detect import detect_schema

REQUIRED_CANON = ["description", "priority", "sentiment"]

def rating_to_sentiment(v) -> Optional[str]:
    try:
        x = float(v)
    except Exception:
        return None
    if x <= 2:
        return "negative"
    if x >= 4:
        return "positive"
    return "neutral"

def _from_raw(raw_csv: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    mapping = detect_schema(list(df.columns))
    desc = mapping.get("description")
    pri = mapping.get("priority")
    sat = mapping.get("satisfaction")
    missing = [k for k, v in {"description": desc, "priority": pri, "satisfaction": sat}.items() if not v]
    if missing:
        raise ValueError(f"Raw CSV missing required fields: {missing}. Columns found: {list(df.columns)}")
    out = pd.DataFrame({
        "description": df[desc].astype(str).str.strip(),
        "priority": df[pri].astype(str).str.strip(),
        "sentiment": df[sat].apply(rating_to_sentiment)
    })
    out = out.dropna(subset=REQUIRED_CANON).reset_index(drop=True)
    return out

def load_preprocessed_or_raw(pre_csv: Optional[str], raw_csv: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    Return (df, source), where df has canonical columns [description, priority, sentiment]
    and source is a string describing which path was used.
    """
    if pre_csv and os.path.isfile(pre_csv):
        df = pd.read_csv(pre_csv)
        missing = [c for c in REQUIRED_CANON if c not in df.columns]
        if missing:
            raise ValueError(f"Preprocessed CSV missing {missing}: {pre_csv}")
        df = df.dropna(subset=REQUIRED_CANON).reset_index(drop=True)
        return df, f"preprocessed:{pre_csv}"
    if raw_csv and os.path.isfile(raw_csv):
        return _from_raw(raw_csv), f"raw:{raw_csv}"
    raise FileNotFoundError("No CSV found (neither preprocessed nor raw).")

def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise labels to consistent casing (en-UK)."""
    df = df.copy()
    df["priority"] = df["priority"].replace({"Critical": "High"})
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()  # 'positive','neutral','negative'
    return df

def apply_basic_hygiene(df: pd.DataFrame, min_len: int = 10, dedup: bool = True) -> pd.DataFrame:
    """Drop too-short descriptions and duplicates."""
    df = df[df["description"].astype(str).str.len() >= min_len]
    if dedup:
        df = df.drop_duplicates(subset=["description"])
    return df.reset_index(drop=True)
