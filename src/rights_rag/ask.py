"""Ask a question from the command line.

uv run python -m rights_rag.ask "כמה ימי חופשה מגיעים לי?"
uv run python -m rights_rag.ask "..." --json
"""

import argparse
import json
from urllib.parse import unquote

STATUS = {
    "answered": "",
    "abstained": "(no answer in the sources)",
    "blocked": "(blocked: quotes could not be verified)",
}


def print_response(r: dict) -> None:
    print(r["answer"], STATUS[r["status"]], sep="\n" if STATUS[r["status"]] else "")
    print("\nSources:")
    for s in r["sources"]:
        print(f"  [{s['n']}] {s['section']}\n      {unquote(s['url'])}")
    if r["citations"]:
        ok = sum(c["valid"] for c in r["citations"])
        print(f"\nCitations verified: {ok}/{len(r['citations'])}")
    print(f"\n{r['seconds']}s, ${r['cost_usd']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--json", action="store_true", help="print the full response as JSON"
    )
    args = parser.parse_args()

    from rights_rag.service import default_answerer

    r = default_answerer().ask(args.question)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print_response(r)


if __name__ == "__main__":
    main()
