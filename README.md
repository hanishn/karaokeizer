# Karaokeizer

Karaokeizer is a local, evidence-driven pipeline for turning legitimately acquired audio into:

- a checkpointed instrumental album produced with Demucs;
- word-timed lyric candidates produced with Whisper;
- pitch, rhythm, key, measure, and notation data;
- a self-contained local HTML karaoke player; and
- machine-readable sonic, transcript, alignment, pitch, and rendered-UI audit evidence.

The project is deliberately conservative. Generated lyrics are drafts, model agreement is not proof, and malformed timing is quarantined instead of being rendered. Source audio is never committed or deleted.

> **Alpha software:** use copies of purchased audio. Review generated lyrics and separation quality before relying on the output.

## What makes the pipeline repeatable

Each stage consumes a manifest and writes versioned JSON evidence. Candidate transcripts live in immutable directories and are promoted only after their required gates pass. A failed region can be retranscribed without rerunning an entire album.

```text
purchased audio
  -> hashed source manifest
  -> Demucs stems
  -> encoded instrumental album
  -> Whisper transcript candidates
  -> structural audit and targeted remediation
  -> pitch/rhythm/key analysis
  -> local HTML player
  -> album and Playwright audit ledgers
```

See [the QA and promotion model](docs/qa-pipeline.md) for the full lifecycle.
For a complete release run, recovery rules, and review responsibilities, use
the [operator guide](docs/operator-guide.md).

## Requirements

- Python 3.11 or 3.12
- FFmpeg and FFprobe on `PATH`
- Enough disk space for lossless source audio, stems, model caches, and candidates
- A CPU works; a supported GPU makes Demucs and Whisper substantially faster

The project does not download music. You supply audio you are authorized to process.

## Installation

```powershell
git clone https://github.com/hanishn/karaokeizer.git
cd karaokeizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements-production-windows-py312.lock
python -m pip install --no-deps --no-build-isolation .
playwright install chromium
```

The production lock targets Windows and Python 3.12, the primary supported
processing environment.

For development on macOS or Linux, use the cross-platform development lock:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation .
playwright install chromium
```

Install FFmpeg with the platform package manager (`brew install ffmpeg` or
`sudo apt install ffmpeg`). Development and CI are locked on Windows and Linux.
A full non-Windows production dependency set is not yet claimed as bit-for-bit
reproducible; live-resolved optional groups remain available for expert use.

## First album

1. Copy [the example configuration](examples/album-config.json) outside the repository and edit its paths and filename pattern.
2. Build and independently verify the instrumental album:

```powershell
karaokeizer-album all --config C:\path\to\album-config.json
```

3. Transcribe separated vocal stems into a new candidate directory:

```powershell
karaokeizer-transcribe `
  --album-manifest C:\path\to\output\pipeline-manifest.json `
  --stem-root C:\path\to\work `
  --output-dir C:\path\to\work\transcripts\candidate-001 `
  --model large-v3
```

4. Audit before building anything user-facing:

```powershell
karaokeizer-audit-transcripts C:\path\to\work\transcripts\candidate-001 `
  --output C:\path\to\work\transcripts\candidate-001\timing-audit.json `
  --strict
```

5. Resolve failures according to [the QA pipeline](docs/qa-pipeline.md), then build and serve the player:

First build the pinned SSLI formant integration from a pristine checkout, then
create the safety allowlist for that reviewed browser distribution:

```powershell
git clone https://github.com/hanishn/SuperSynthLabInstrument.git C:\path\to\ssli-source
git -C C:\path\to\ssli-source checkout 882a1e061d75c3db95b6e63fd8ab9980102e24d4
karaokeizer-prepare-ssli C:\path\to\ssli-source C:\path\to\ssli-distribution
karaokeizer-manifest-ssli C:\path\to\ssli-distribution
```

The command rejects symlinks/junctions, hidden files, unexpected types, and
oversized trees; rerun it whenever the distribution intentionally changes. The
distribution must implement Karaokeizer's timestamped formant scheduling
contract; immediate `noteOn`/timer-based releases are deliberately rejected.
See [the SSLI integration contract](docs/ssli-integration.md).

```powershell
karaokeizer-build-player `
  --album-manifest C:\path\to\output\pipeline-manifest.json `
  --transcripts C:\path\to\work\transcripts\promoted `
  --stem-root C:\path\to\work `
  --media-root C:\path\to\output `
  --ssli-root C:\path\to\ssli-distribution `
  --analysis-overrides C:\path\to\analysis-overrides.json `
  --output-dir C:\path\to\work\rendered-candidate-001

karaokeizer-serve --root C:\path\to\work\rendered-candidate-001 --port 8003
```

`--output-dir` must be a new or empty candidate directory. The builder never
silently replaces the canonical player. After the album, transcript-consensus,
lyric-review, and rendered-browser ledgers pass, use `karaokeizer-promote` to
copy the exact hash-bound candidate into the canonical directory. See the
complete command sequence in [the QA guide](docs/qa-pipeline.md).

The server prints the local URL. File URLs are intentionally avoided because browsers impose inconsistent media and iframe restrictions on `file://` pages.

## Verification gates

| Gate | Examples of failures caught |
|---|---|
| Sonic | Duration/layout mismatch, clipping, weak stem reconstruction, leakage |
| Transcript structure | Collapsed words, duplicate starts, overlaps, words outside segments |
| Lexical | Disagreement with an authorized lyric reference or independent transcript |
| Alignment and pitch | Lyrics over silence, uncovered vocal activity, missing note mappings |
| Rendered player | Browser exceptions, word collisions, overflow, high-risk timestamp regressions |

Passing structural timing does **not** mean that generated words are lyrically correct. Supply an authorized lyric reference or perform a listening review before marking lyrics verified.

Likewise, stem reconstruction alone cannot prove that the accompaniment is vocal-free. The album audit requires an independent per-track separation review using the format in [`examples/separation-review.json`](examples/separation-review.json); pass it with `--separation-review`.

## Repository hygiene

The `.gitignore` excludes audio, model files, generated transcript/data payloads, work directories, and screenshots. Before committing, still inspect `git status`—large media has a remarkable talent for finding the one filename pattern nobody anticipated.

## Development

```powershell
python -m pip install -e ".[dev]"
ruff check .
pytest
```

Tests use synthetic fixtures and never require the private album used during initial development. Integration tests requiring Demucs, Whisper models, FFmpeg, or browser binaries should be run explicitly in an appropriately provisioned environment.

## Security and privacy

Read [SECURITY.md](SECURITY.md). Keep purchased media, generated stems, transcripts, and lyric references out of commits unless you have the rights to redistribute them.

## License

Karaokeizer is licensed under the [Apache License 2.0](LICENSE). This repository
includes an Apache-2.0 modification patch for the pinned SSLI formant engine and
AudioWorklet; its attribution is preserved in [NOTICE](NOTICE). The full SSLI
distribution, media, and models are not bundled, and their own licenses and
redistribution terms continue to apply.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes to validation thresholds or promotion rules must include evidence and regression tests; a deadline is not evidence.
