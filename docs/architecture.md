# Architecture

Karaokeizer is a set of narrow command-line stages joined by explicit JSON artifacts. Stages do not share hidden process state.

## Core artifacts

- **Album configuration:** user-authored paths, filename pattern, encoding settings, and Demucs model.
- **Pipeline manifest:** immutable source/output hashes, tool versions, audio properties, and stem paths.
- **Transcript candidate:** model identity, source-stem hash, segments, words, probabilities, and timing.
- **QA ledger:** gate results, failures, warnings, metrics, and remediation regions.
- **Karaoke data:** detected key, rhythm, pitch contour, word-note binding, and rendering metadata.
- **Player:** generated HTML that embeds only compact metadata and references local encoded audio.

## Candidate isolation

A candidate directory is a transaction boundary. A model tier writes into a fresh directory. A replacement region is validated before it is merged; a rejected region leaves the prior candidate unchanged. Promotion is a separate, reviewable operation.

## Dependency boundaries

Large dependencies are optional:

- Demucs is imported only by the separation stage.
- Whisper backends are imported only by transcription/remediation stages.
- Playwright is imported only by rendered-player QA.

This keeps static analysis and lightweight unit tests independent of model installations.

## Failure philosophy

Commands return nonzero when a required gate fails and still write a report whenever practical. A completed process is not automatically a passing artifact. Instrumental tracks are classified explicitly instead of being forced through speech recognition.
