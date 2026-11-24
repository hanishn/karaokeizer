# Operator guide

This guide describes the release path, not a quick preview. Karaokeizer fails
closed when an input changes: regenerate the affected candidate and its
downstream evidence rather than editing a ledger to make it pass.

## State model

| State | May be edited? | Meaning |
|---|---:|---|
| Purchased source | No | Operator-owned input; the pipeline never deletes it |
| Separation work | Only with `separate --force` | Demucs stems plus a source/model/output checkpoint |
| Packaged album | Only with `package --force` | FLAC plus an encoding recipe checkpoint |
| Transcript candidate | No | One stem hash and one complete decoder/model identity |
| Rendered candidate | No | Fresh self-contained player tree: HTML, data, audio, and SSLI |
| Canonical player | Only through promotion | Exact candidate that passed all bound ledgers |

`--force` is destructive only to a named generated stage; it never authorizes
source deletion. `--resume` continues an identified candidate. A fresh
candidate directory is preferred whenever model, settings, or evidence change.

## Platform-neutral setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation .
python -m playwright install chromium
```

macOS/Linux shell:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation .
python -m playwright install chromium
```

Install FFmpeg/FFprobe separately and confirm both are on `PATH`. On the primary
Windows/Python 3.12 processing platform, use
`requirements-production-windows-py312.lock` instead of the dev lock to install
Demucs, both Whisper backends, and browser QA with hashes. For `faster-whisper`,
supply a full immutable 40-character model commit; mutable tags are rejected.

## Release sequence

Use absolute paths in commands but keep all generated directories outside the
repository.

1. Run `karaokeizer-album all --config album-config.json`. If a stem or package
   recipe changed, the checkpoint refuses reuse; inspect the difference, then
   rerun the named stage with `--force` if intentional.
2. Create two transcript candidates with different decoder identities. Keep the
   directories immutable.
3. Run the structural audit and regional remediation. Remediation accepts only
   a QA ledger bound to the same manifest, transcript, and vocal-stem hashes.
4. Run transcript consensus. Both global and repaired-region disagreement must
   pass; identical candidates are rejected as non-independent.
5. Complete `separation-review.json`, `analysis-overrides.json`, and
   `lyric-review.json`. Each review names its reviewer and method. Separation
   rows contain the exact source/vocal/accompaniment hashes reviewed.
6. Build into a new rendered-candidate directory. The builder copies packaged
   audio and the manifest-verified SSLI distribution, making the candidate
   preview and canonical tree identical. Run `karaokeizer-manifest-ssli` on a
   reviewed prebuilt distribution first; linked, hidden, unexpected, or
   oversized files are rejected.
7. Serve that candidate with `karaokeizer-serve`. Run the album and Chromium
   audits. The visual audit refuses a URL whose response bytes differ from the
   local player file and records failed browser/media requests.
8. Run `karaokeizer-promote` with all ledgers. It validates schemas, track-set
   completeness, manifest identity, transcript/review hashes, audio, HTML,
   karaoke data, and SSLI files before staging the complete tree.

## Evidence ownership

Machine output is evidence, not approval. The following reviews require a
named human:

- separation: listen to source, vocal stem, and accompaniment for every track;
- lyrics: compare every word to an authorized reference or careful listening;
- analysis: verify key, meter, and downbeat phase over multiple measures.

Copy the files in `examples/`, replace every placeholder, and set `pass` only
after all rows are complete. Never reuse a review after its bound hash changes.

## Failure and recovery

| Failure | Recovery |
|---|---|
| Stale separation checkpoint | Re-run `separate --force`, then regenerate every downstream artifact |
| Stale package checkpoint | Re-run `package --force`; rebuild player and evidence |
| Transcript provenance mismatch | Transcribe the manifest's current vocal stem into a fresh directory |
| Consensus disagreement | Listen, repair bounded regions, and compare independent candidates again |
| Separation hash mismatch | Re-listen to the exact current stems and write a new review |
| Visual URL/file mismatch | Stop stale server, serve the candidate directory, rerun Chromium audit |
| Browser/media request failure | Fix candidate asset paths; do not promote a silent preview |
| Promotion copy failure | Canonical remains unchanged; inspect the staging error and retry |
| Post-swap operational problem | Restore the `previous_canonical_backup` recorded in the receipt |

Promotion retains the previous canonical directory as a timestamped sibling
backup. Verify the new canonical player before removing that backup manually.
The tool intentionally does not automate backup deletion.

## Release checklist

- [ ] Source and package checkpoints match current inputs.
- [ ] Two genuinely independent transcript candidates exist.
- [ ] Structural, consensus, sonic, alignment, pitch, and visual ledgers pass.
- [ ] Separation, lyric, key, and meter reviews are named and hash-bound.
- [ ] Candidate playback, seeking, track change, and guide voice work locally.
- [ ] Repository hygiene, lint, unit tests, browser test, and wheel smoke pass.
- [ ] Promotion receipt lists every artifact and any previous backup.
