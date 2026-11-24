# Karaokeizinator album-scale QA and remediation pipeline

## Objective

Turn a purchased album plus separated stems into a karaoke player whose audio, lyrics, timing, pitch, and rendered presentation are independently checked before promotion. Candidate artifacts never overwrite canonical artifacts until all required gates pass.

## Artifact lifecycle

1. **Source** — immutable purchased audio and its hashes.
2. **Stems** — versioned Demucs output with reconstruction evidence.
3. **Transcript candidate** — isolated by engine/model/settings.
4. **Analysis candidate** — pitch, rhythm, word-note binding, and confidence.
5. **Rendered candidate** — HTML player plus screenshots and browser results.
6. **Canonical** — promoted only after machine gates and lyric review pass.

Every stage is resumable and content-addressed. A changed source, stem, model, transcript, or analysis setting invalidates downstream checkpoints.

## Required gates

### 1. Sonic/stem gate

- Source, vocal stem, and accompaniment have matching duration, sample rate, and channel layout.
- No output clipping.
- Gain-normalized vocal-plus-accompaniment reconstruction error is measured and bounded.
- Vocal/accompaniment correlation is recorded as leakage evidence.
- Instrumental tracks are detected rather than forced through lyric generation.

Tool: `karaokeizer-audit-album`

### 2. Transcript structure gate

- Positive word duration.
- Strictly monotonic word starts.
- No duplicate starts inside a segment.
- No material word overlap.
- Words remain inside segment bounds.
- Unexpected writing systems, repetition, low confidence, and implausible word density are flagged.

Malformed segments are never rendered. They become remediation regions.

Tools: `karaokeizer-audit-transcripts`, validation inside `karaokeizer-build-player`

### 3. Lyrical gate

Preferred order of evidence:

1. User-supplied official lyrics or liner-note text.
2. A second independent transcription pass with a different model/backend.
3. Human listening review for every disagreement and every low-confidence word.

Model agreement is not proof of correctness; it only reduces the human-review surface. Without an authorized reference, the transcript remains explicitly unverified.

Tool: `karaokeizer-compare-transcripts`

### 4. Sonic alignment and pitch gate

- Vocal activity is substantially covered by lyric intervals.
- Lyrics do not primarily cover silence.
- Each rendered lyric has a directly detected or explicitly estimated pitch event.
- Estimated-pitch fraction and rejected-segment count are visible in the ledger.
- Note onset, sustain, syllable attacks, and release share one timeline.

Tools: `karaokeizer-audit-album`, `karaokeizer-build-player`

### 5. Rendered visual gate

- Every lyric-bearing track is sampled at the beginning, middle, and end of its vocal content.
- No same-lane word collisions.
- No horizontal page overflow.
- No browser exceptions.
- Screenshots are retained as evidence.
- Known high-risk regions from transcript remediation receive extra samples.

Tool: `karaokeizer-audit-player`

## Automatic remediation loop

1. Run the transcript audit and merge adjacent defective segments into padded regions.
2. Re-transcribe only those regions using a fast English model (`base.en` by default), greedy decoding, `condition_on_previous_text=False`, and word timestamps.
3. Validate the replacement *before* merging. The merge is atomic: collapsed words, duplicate starts, non-monotonic timing, out-of-segment words, punctuation-only artifacts, and empty replacements are rejected.
4. Escalate any region that still fails—or materially disagrees with an independent transcript/reference—to `large-v3` with beam search.
5. Rebind the revised words to cached pitch-contour frames.
6. Rebuild the player.
7. Run album, browser, and visual audits again.
8. Repeat until the region passes or mark it `requires_human_lyric_review`.

Do not repair timing by copying a different chorus unless it is an emergency preview explicitly labeled as estimated. A fresh vocal-region pass is the canonical remediation.

Each escalation tier writes to a fresh immutable candidate directory; directory reuse requires an explicit `--resume`. Tracks with no reliable vocal activity are explicitly classified and receive an empty lyric payload. Whisper hallucinations on instrumental passages are never promoted as lyrics.

## Promotion rules

A track may be promoted only when:

- sonic, timing, alignment, pitch, and visual gates pass;
- no malformed segment is present in the rendered payload;
- lexical disagreements have been resolved against an authorized reference or by listening review;
- the ledger records model, hashes, commands, and evidence paths.

Album promotion requires every track to be promoted or explicitly classified as instrumental/no-reliable-vocals.

## Recommended evidence layout

- `karaoke/album-qa-ledger.json` — combined sonic, lyric, alignment, and pitch audit.
- `karaoke/transcript-timing-audit.json` — transcript structural failures.
- `karaoke/player-visual-audit.json` — rendered layout/browser audit.
- `karaoke/visual-audit-evidence/` — retained screenshots from every track.
- `karaoke/candidates/<engine-model-run>/` — immutable, quarantined transcript candidates.

## Reusable command sequence

```powershell
karaokeizer-transcribe --album-manifest <manifest> --stem-root <work-root> --output-dir <candidate-dir> --model large-v3 --backend openai --beam-size 8 --force
karaokeizer-audit-transcripts <candidate-dir> --output <candidate-dir>/timing-audit.json --strict
karaokeizer-compare-transcripts --primary <canonical-dir> --candidate <candidate-dir> --output <qa-dir>/transcript-consensus.json
karaokeizer-build-player --album-manifest <manifest> --transcripts <promoted-transcripts> --stem-root <work-root> --media-root <packaged-audio-root> --ssli-root <local-ssli-distribution> --analysis-overrides <reviewed-analysis-overrides> --output-dir <fresh-rendered-candidate>
karaokeizer-audit-album --album-manifest <manifest> --source-root <source-root> --stem-root <work-root> --transcripts <promoted-transcripts> --karaoke-data <fresh-rendered-candidate>/karaoke/data --separation-review <separation-review.json> --output <qa-dir>/album-qa-ledger.json
karaokeizer-audit-player --url <served-player-url> --player <fresh-rendered-candidate>/<player.html> --output <qa-dir>/player-visual-audit.json --evidence-dir <qa-dir>/visual-evidence
karaokeizer-promote --candidate-dir <fresh-rendered-candidate> --canonical-dir <canonical-root> --transcripts <promoted-transcripts> --album-ledger <qa-dir>/album-qa-ledger.json --visual-ledger <qa-dir>/player-visual-audit.json --consensus-ledger <qa-dir>/transcript-consensus.json --lyric-review <qa-dir>/lyric-review.json --analysis-review <analysis-overrides.json> --receipt <qa-dir>/promotion-receipt.json
```

The command sequence is deliberately album-agnostic; paths and manifest content define the album.
