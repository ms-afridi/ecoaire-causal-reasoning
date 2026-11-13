import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "precomputed_cache.json"


def cache_key(paper_id: str, abstract_text: str) -> str:
    return f"{paper_id}:{hashlib.sha256(abstract_text.encode('utf-8')).hexdigest()[:16]}"


def load_cache(path: Path = CACHE_PATH) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_precomputed(paper_id: str, abstract_text: str, path: Path = CACHE_PATH) -> Optional[Dict[str, Any]]:
    entry = load_cache(path).get(cache_key(paper_id, abstract_text))
    if not entry or "result" not in entry:
        return None
    result = dict(entry["result"])
    result["source"] = "precomputed"
    return {"result": result, "meta": entry.get("meta", {})}


def put_precomputed(paper_id: str, abstract_text: str, result: Dict[str, Any], path: Path = CACHE_PATH) -> None:
    cache = load_cache(path)
    cache[cache_key(paper_id, abstract_text)] = {
        "result": result,
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "model": result.get("model_name"),
            "judge_model": result.get("judge_model"),
        },
    }
    Path(path).write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
