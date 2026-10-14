# Local retrieval index

`chunks.jsonl` contains compact text chunks with stable IDs, relative source paths, headings, and text. `manifest.json` records document hashes.

Rebuild after changing notes or updating repositories:

```bash
python3 tools/build_index.py
```

The index intentionally includes our notes and each repository's root README, not every source file. Use `rg` for exact source-code search and the JSONL index for conceptual retrieval.
