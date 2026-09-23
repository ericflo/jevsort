"""Loading items and datasets from JSON / JSONL / CSV / text."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .sorter import Item


def _item_from(d: dict, n: int) -> Item:
    d = dict(d)
    iid = str(d.pop("id", None) or d.get("key") or f"item{n + 1}")
    if "text" in d:
        text = str(d.pop("text"))
    else:
        title = d.pop("title", "")
        body = d.pop("abstract", None) or d.pop("body", None) or d.pop("description", None) or ""
        text = f"{title}\n{body}".strip() if title else str(body)
    return Item(iid, text, meta=d)


def load_items(path) -> tuple[list[Item], dict]:
    """Load items. Returns (items, extra) where extra may contain ``objective``."""
    p = Path(path)
    extra: dict = {}
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    elif p.suffix == ".csv":
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            rows = data.get("items", [])
            extra = {k: v for k, v in data.items() if k != "items"}
        else:
            rows = data
    else:  # plain text: one item per non-empty line
        rows = [{"text": line.strip()} for line in p.read_text().splitlines() if line.strip()]
    items = [_item_from(r, n) if isinstance(r, dict) else Item(f"item{n + 1}", str(r)) for n, r in enumerate(rows)]
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique")
    return items, extra


def labels_of(items: list[Item], key: str):
    """Ground-truth label vector for ``key`` (e.g. a dimension or "overall")."""
    import numpy as np

    out = []
    for it in items:
        lab = it.meta.get("labels", {})
        if key == "overall" and "overall" not in lab:
            vals = [float(v) for v in lab.values()]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        else:
            out.append(float(lab.get(key, float("nan"))))
    return np.array(out)
