# Contributing

## Development setup

Create an isolated environment and install the editable development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
playwright install chromium
```

Run the fast checks before opening a pull request:

```powershell
ruff check .
pytest
```

## Change rules

- Never add copyrighted album audio, stems, model caches, generated lyric corpora, or machine-specific absolute paths.
- Treat source audio as immutable. Destructive cleanup does not belong in the processing pipeline.
- Write candidates to new directories. Reuse must be explicit and provenance-preserving.
- Do not weaken a QA threshold merely to make a fixture pass. Explain and test any threshold change.
- Keep model output labeled as unverified until it is checked against an authorized reference or by listening review.
- Add a regression test for every bug fix when a deterministic test is possible.

## Pull-request evidence

Describe the affected pipeline stage, the command used to verify it, and the resulting machine-readable report. UI changes should include Playwright results and screenshots at ordinary and known-risk timestamps.
