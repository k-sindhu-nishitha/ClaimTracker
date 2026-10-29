"""Terminal demo.   python -m claimtrackr.cli index | demo [-v] | health"""
from __future__ import annotations

import argparse
import json
import logging

from .config import ROOT, get_settings
from .embeddings import get_embedder
from .models import ClaimInput
from .pdf_utils import extract_text
from .processor import ClaimProcessor
from .rag import build_index


def main() -> None:
    parser = argparse.ArgumentParser(prog="claimtrackr")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="rebuild the FAISS index from documents/")
    demo = sub.add_parser("demo", help="run the sample claims through the full pipeline")
    demo.add_argument("-v", "--verbose", action="store_true", help="print every pipeline step")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()

    if args.cmd == "index":
        idx = build_index(settings, get_embedder(settings))
        print(f"Indexed {len(idx.chunks)} chunks with {idx.embedder_name} -> {settings.index_dir}")
        return

    processor = ClaimProcessor(settings)
    print(f"Engine: LLM={processor.backend.name} | embeddings={processor.embedder.name} | chunks={len(processor.index.chunks)}\n")
    cases = json.loads((ROOT / "sample_data" / "demo_cases.json").read_text(encoding="utf-8"))
    ok_count = 0
    print(f"{'CASE':<38}{'EXPECTED':<15}{'ACTUAL':<15}RESULT")
    for case in cases:
        result = processor.process(ClaimInput(**case["claim"]), extract_text(str(ROOT / case["bill"])))
        ok = result.verdict == case["expected"]
        ok_count += ok
        print(f"{case['label']:<38}{case['expected']:<15}{result.verdict:<15}{'OK' if ok else 'MISMATCH'}")
        if args.verbose:
            for s in result.steps:
                print(f"    [{s.status.upper():<4}] {s.name}: {s.detail}")
            print()
    print(f"\nMatched {ok_count}/{len(cases)} expected verdicts")


if __name__ == "__main__":
    main()
