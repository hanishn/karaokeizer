# Security and privacy

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose local files or execute commands. Use GitHub's private vulnerability-reporting feature when it is enabled for the repository.

## Trust boundaries

Karaokeizer processes untrusted filenames, metadata, manifests, model output, and lyric text. It invokes FFmpeg, FFprobe, and Demucs without a shell and passes arguments as discrete values. Keep that property intact.

The local server binds to `127.0.0.1` by default. Do not expose it to another interface unless you understand that generated players and media may then be reachable by other machines.

## Sensitive and copyrighted data

Purchased audio, separated stems, transcripts, lyric references, screenshots, and model caches can be sensitive or redistribution-restricted. They are excluded by the repository defaults, but users remain responsible for reviewing commits and complying with applicable licenses and law.
