import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groq import Groq  # noqa: E402

from src.cache import put_precomputed  # noqa: E402
from src.pipeline import run_extraction  # noqa: E402


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Set GROQ_API_KEY first.")
    client = Groq(api_key=api_key)
    papers = json.loads((ROOT / "data" / "sample_papers.json").read_text(encoding="utf-8"))
    for paper in papers:
        print(f"Running {paper['id']} ...")
        result = run_extraction(client, paper["id"], paper["title"], paper["abstract_text"])
        put_precomputed(paper["id"], paper["abstract_text"], result)
        ev = result["evaluation"]
        print(
            f"  provenance exact {ev['provenance_edges_exact']}/{ev['provenance_edges_total']}"
            f" | rigor {ev['causal_rigor_score']} | completeness {ev['contextual_completeness']}"
        )
    print("Cache written to data/precomputed_cache.json")


if __name__ == "__main__":
    main()
