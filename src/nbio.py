from pathlib import Path
import uuid
import nbformat

__all__ = [
    "read_nb",
    "write_nb",
    "ensure_cell_ids",
    "find_cell_index",
]


def read_nb(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ensure_cell_ids(nb)
    return nb


def write_nb(path: str, nb: dict) -> None:
    ensure_cell_ids(nb)
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def ensure_cell_ids(nb: dict) -> None:
    for c in nb.get("cells", []):
        cid = c.get("id") or c.get("metadata", {}).get("id")
        if not cid:
            cid = uuid.uuid4().hex[:8]
            c["id"] = cid
        c.setdefault("metadata", {}).setdefault("id", c["id"])


def find_cell_index(nb: dict, cell_id: str) -> int | None:
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("id") == cell_id or c.get("metadata", {}).get("id") == cell_id:
            return i
    return None


