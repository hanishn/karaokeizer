"""Resumable and verifiable local instrumental-album pipeline.

Stages:
  separate  - create Demucs vocal/no-vocal stems, one checkpointed track at a time
  package   - encode/tag FLAC files and write playlists plus a provenance manifest
  verify    - independently re-hash and probe every source/output file
  all       - run all three stages

The script never deletes source audio. Work-file cleanup is intentionally separate.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import regex

from .html_safety import json_for_script
from .paths import safe_component

SCHEMA = "instrumental-album-pipeline/v1"
PACKAGE_RECIPE_VERSION = "flac-s32-level12-attached-cover/v1"


def command(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an external tool and echo its exact command for reproducibility."""
    print("+", subprocess.list2cmdline(args), flush=True)
    return subprocess.run(args, check=True, text=True, capture_output=capture)


def sha256(path: Path) -> str:
    """Hash a file incrementally for manifests and checkpoint identities."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_version(executable: str) -> str:
    """Read the first line of an FFmpeg-family tool's version output."""
    result = command([executable, "-version"])
    return result.stdout.splitlines()[0].strip()


def audio_probe(path: Path, ffprobe: str) -> dict:
    """Return normalized audio properties used by independent verification."""
    result = command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,sample_fmt,codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "duration_seconds": round(float(payload["format"]["duration"]), 6),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "sample_format": stream["sample_fmt"],
        "codec": stream["codec_name"],
    }


def load_config(path: Path) -> dict:
    """Load defaults and reject incomplete or unsafe configuration values."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["artist", "album", "source_dir", "output_dir", "work_dir"]
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")
    config.setdefault("year", "")
    config.setdefault("genre", "")
    config.setdefault("source_glob", "*.wav")
    config.setdefault("track_regex", r" - (?P<number>\d{2}) (?P<title>.+)\.wav$")
    config.setdefault("cover", "cover.jpg")
    config.setdefault("demucs_model", "htdemucs")
    config.setdefault("album_suffix", " (Instrumental)")
    safe_component(config["album"], field="album")
    return config


def source_tracks(config: dict) -> list[dict]:
    """Discover source tracks while bounding user-configured regex runtime."""
    source_dir = Path(config["source_dir"])
    expression = config["track_regex"]
    if len(expression) > 1_000:
        raise ValueError("track_regex may not exceed 1,000 characters")
    pattern = regex.compile(expression, regex.IGNORECASE)
    tracks = []
    for path in sorted(source_dir.glob(config["source_glob"])):
        try:
            match = pattern.search(path.name, timeout=0.05)
        except TimeoutError as error:
            raise ValueError("track_regex exceeded the 50 ms safety limit") from error
        if not match:
            raise ValueError(f"Source filename does not match track_regex: {path.name}")
        tracks.append(
            {
                "number": int(match.group("number")),
                "title": match.group("title"),
                "source": path,
            }
        )
    tracks.sort(key=lambda row: row["number"])
    expected = list(range(1, len(tracks) + 1))
    actual = [row["number"] for row in tracks]
    if not tracks or actual != expected:
        raise ValueError(f"Track numbers must be contiguous from 1: {actual}")
    return tracks


def stem_dir(config: dict, source: Path) -> Path:
    """Return the deterministic Demucs output directory for a source track."""
    return Path(config["work_dir"]) / config["demucs_model"] / source.stem


def package_recipe(config: dict, row: dict, stem: Path, cover: Path, ffmpeg_version: str) -> dict:
    """Describe every material input and setting used to encode one FLAC."""

    return {
        "schema": "karaokeizer-package-checkpoint/v1",
        "recipe_version": PACKAGE_RECIPE_VERSION,
        "ffmpeg_version": ffmpeg_version,
        "accompaniment_sha256": sha256(stem),
        "cover_sha256": sha256(cover),
        "metadata": {
            "artist": config["artist"],
            "album": f"{config['album']}{config['album_suffix']}",
            "title": f"{row['title']} (Instrumental)",
            "track": row["number"],
            "year": config["year"],
            "genre": config["genre"],
        },
    }


def separation_recipe(config: dict, source: Path) -> dict:
    """Return the provenance identity required to resume a Demucs checkpoint."""

    return {
        "schema": "karaokeizer-demucs-checkpoint/v1",
        "source_sha256": sha256(source),
        "demucs_version": importlib.metadata.version("demucs"),
        "model": config["demucs_model"],
        "arguments": ["--two-stems=vocals", "--float32"],
    }


def separate(config: dict, python: str, *, force: bool = False) -> None:
    """Create stems, resuming only when provenance and output hashes match."""
    work_dir = Path(config["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    tracks = source_tracks(config)
    for index, row in enumerate(tracks, 1):
        destination = stem_dir(config, row["source"])
        vocal = destination / "vocals.wav"
        accompaniment = destination / "no_vocals.wav"
        checkpoint_path = destination / "karaokeizer-checkpoint.json"
        recipe = separation_recipe(config, row["source"])
        if vocal.exists() and accompaniment.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else None
            valid = bool(
                checkpoint
                and checkpoint.get("recipe") == recipe
                and checkpoint.get("outputs", {}).get("vocals.wav") == sha256(vocal)
                and checkpoint.get("outputs", {}).get("no_vocals.wav") == sha256(accompaniment)
            )
            if valid:
                print(f"[{index}/{len(tracks)}] verified checkpoint exists: {row['title']}")
                continue
            if not force:
                raise RuntimeError(
                    f"stale or unverifiable stem checkpoint for {row['title']!r}; "
                    "use a fresh work_dir or rerun with --force"
                )
        print(f"[{index}/{len(tracks)}] separating: {row['title']}")
        command(
            [
                python,
                "-m",
                "demucs",
                "--two-stems=vocals",
                "-n",
                config["demucs_model"],
                "--float32",
                "--out",
                str(work_dir),
                str(row["source"]),
            ],
            capture=False,
        )
        if not vocal.exists() or not accompaniment.exists():
            raise RuntimeError(f"Demucs did not produce both stems for {row['source'].name}")
        checkpoint_path.write_text(
            json.dumps({
                "recipe": recipe,
                "outputs": {"vocals.wav": sha256(vocal), "no_vocals.wav": sha256(accompaniment)},
            }, indent=2) + "\n",
            encoding="utf-8",
        )


def output_filename(row: dict) -> str:
    """Return a portable output name derived from a validated track title."""
    title = safe_component(row["title"], field=f"track {row['number']} title")
    return f"{row['number']:02d} - {title} (Instrumental).flac"


def write_player(output: Path, artist: str, album: str, tracks: list[dict]) -> None:
    """Generate the simple instrumental player with HTML-safe metadata."""
    payload = json_for_script([{"file": row["output_file"], "title": row["title"]} for row in tracks])
    safe_artist = html.escape(artist)
    safe_album = html.escape(album)
    html_document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_artist} — {safe_album} (Instrumental)</title><style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#080b1b;color:#f5f2ff}}
main{{width:min(920px,92vw);margin:2rem auto;display:grid;grid-template-columns:minmax(220px,340px) 1fr;gap:2rem}}
img{{width:100%;border-radius:12px;box-shadow:0 16px 50px #0009}}audio{{width:100%;margin:.75rem 0 1rem}}
button{{width:100%;text-align:left;color:inherit;background:transparent;border:0;padding:.6rem;border-radius:7px;cursor:pointer}}
button:hover,li.active button{{background:#7c3aed44}}@media(max-width:700px){{main{{grid-template-columns:1fr}}}}
</style></head><body><main><section><img src="cover.jpg" alt="Album cover"></section><section>
<h1>{safe_album} — Instrumental</h1><p>{safe_artist} · locally separated accompaniment · FLAC</p>
<strong id="now">Select a track</strong><audio id="audio" controls preload="metadata"></audio><ol id="list"></ol>
</section></main><script>const tracks={payload};let current=0;const audio=document.getElementById('audio'),list=document.getElementById('list'),now=document.getElementById('now');
function load(i,play=true){{current=(i+tracks.length)%tracks.length;audio.src=tracks[current].file;now.textContent=`${{current+1}}. ${{tracks[current].title}}`;[...list.children].forEach((li,n)=>li.classList.toggle('active',n===current));if(play)audio.play().catch(()=>{{}})}}
tracks.forEach((t,i)=>{{const li=document.createElement('li'),b=document.createElement('button');b.textContent=t.title;b.onclick=()=>load(i);li.appendChild(b);list.appendChild(li)}});audio.addEventListener('ended',()=>{{if(current+1<tracks.length)load(current+1)}});load(0,false);
</script></body></html>"""
    (output / f"{album} - Instrumental.html").write_text(html_document, encoding="utf-8")


def package(config: dict, ffmpeg: str, ffprobe: str, *, force: bool) -> Path:
    """Encode accompaniment stems and write the portable album manifest."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cover = Path(config["source_dir"]) / config["cover"]
    if not cover.exists():
        raise FileNotFoundError(cover)
    shutil.copy2(cover, output / "cover.jpg")
    tracks = source_tracks(config)
    work_root = Path(config["work_dir"]).resolve()
    ffmpeg_version = tool_version(ffmpeg)
    manifest_tracks = []
    for row in tracks:
        stem = stem_dir(config, row["source"]) / "no_vocals.wav"
        vocal = stem.with_name("vocals.wav")
        if not stem.exists() or not vocal.exists():
            raise FileNotFoundError(f"Missing stem pair for {row['source'].name}")
        filename = output_filename(row)
        destination = output / filename
        checkpoint_path = destination.with_suffix(destination.suffix + ".karaokeizer.json")
        recipe = package_recipe(config, row, stem, cover, ffmpeg_version)
        if destination.exists() and not force:
            if not checkpoint_path.exists():
                raise RuntimeError(f"existing package lacks provenance checkpoint: {destination}")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("recipe") != recipe or checkpoint.get("output_sha256") != sha256(destination):
                raise RuntimeError(f"stale packaged output; rerun package with --force: {destination}")
        if force or not destination.exists():
            command(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(stem),
                    "-i",
                    str(cover),
                    "-map",
                    "0:a:0",
                    "-map",
                    "1:v:0",
                    "-c:a",
                    "flac",
                    "-compression_level",
                    "12",
                    "-sample_fmt",
                    "s32",
                    "-c:v",
                    "copy",
                    "-disposition:v:0",
                    "attached_pic",
                    "-metadata",
                    f"artist={config['artist']}",
                    "-metadata",
                    f"album={config['album']}{config['album_suffix']}",
                    "-metadata",
                    f"title={row['title']} (Instrumental)",
                    "-metadata",
                    f"track={row['number']}/{len(tracks)}",
                    "-metadata",
                    f"date={config['year']}",
                    "-metadata",
                    f"genre={config['genre']}",
                    "-metadata",
                    "comment=Instrumental accompaniment separated locally with Demucs v4 from an owner-supplied source.",
                    str(destination),
                ]
            )
            checkpoint_path.write_text(
                json.dumps({"recipe": recipe, "output_sha256": sha256(destination)}, indent=2) + "\n",
                encoding="utf-8",
            )
        source_audio = audio_probe(row["source"], ffprobe)
        output_audio = audio_probe(destination, ffprobe)
        manifest_tracks.append(
            {
                "number": row["number"],
                "title": row["title"],
                "source_file": row["source"].name,
                "source_sha256": sha256(row["source"]),
                "source_audio": source_audio,
                "accompaniment_stem": stem.resolve().relative_to(work_root).as_posix(),
                "accompaniment_sha256": sha256(stem),
                "vocal_stem": vocal.resolve().relative_to(work_root).as_posix(),
                "vocal_sha256": sha256(vocal),
                "output_file": filename,
                "output_sha256": sha256(destination),
                "output_audio": output_audio,
                "package_recipe": recipe,
                "package_checkpoint": checkpoint_path.name,
                "package_checkpoint_sha256": sha256(checkpoint_path),
            }
        )
    playlist_name = f"{config['album']} - Instrumental.m3u8"
    playlist = "#EXTM3U\n" + "\n".join(row["output_file"] for row in manifest_tracks) + "\n"
    (output / playlist_name).write_text(playlist, encoding="utf-8-sig")
    write_player(output, config["artist"], config["album"], manifest_tracks)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "config": {key: value for key, value in config.items() if key not in {"source_dir", "output_dir", "work_dir"}},
        "tools": {
            "python": sys.version.splitlines()[0],
            "demucs": importlib.metadata.version("demucs"),
            "ffmpeg": ffmpeg_version,
            "ffprobe": tool_version(ffprobe),
        },
        "separation": {"model": config["demucs_model"], "mode": "two-stems=vocals", "float32": True},
        "playlist": playlist_name,
        "html_player": f"{config['album']} - Instrumental.html",
        "cover_sha256": sha256(cover),
        "tracks": manifest_tracks,
    }
    path = output / "pipeline-manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def verify(config: dict, ffprobe: str) -> dict:
    """Re-hash and re-probe packaged artifacts independently of encoding."""
    output = Path(config["output_dir"])
    manifest_path = output / "pipeline-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    tracks = manifest["tracks"]
    source_dir = Path(config["source_dir"])
    for row in tracks:
        source = source_dir / row["source_file"]
        destination = output / row["output_file"]
        if not source.exists():
            failures.append(f"missing source: {row['source_file']}")
            continue
        if sha256(source) != row["source_sha256"]:
            failures.append(f"source hash changed: {row['source_file']}")
        if not destination.exists():
            failures.append(f"missing output: {row['output_file']}")
            continue
        if sha256(destination) != row["output_sha256"]:
            failures.append(f"output hash changed: {row['output_file']}")
        checkpoint_path = output / row["package_checkpoint"]
        if not checkpoint_path.exists():
            failures.append(f"package checkpoint missing: {row['output_file']}")
        else:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if sha256(checkpoint_path) != row["package_checkpoint_sha256"]:
                failures.append(f"package checkpoint changed: {row['output_file']}")
            if checkpoint.get("recipe") != row["package_recipe"]:
                failures.append(f"package recipe mismatch: {row['output_file']}")
            if checkpoint.get("output_sha256") != row["output_sha256"]:
                failures.append(f"package checkpoint output mismatch: {row['output_file']}")
        actual = audio_probe(destination, ffprobe)
        source_audio = row["source_audio"]
        if abs(actual["duration_seconds"] - source_audio["duration_seconds"]) > 0.1:
            failures.append(f"duration mismatch: {row['output_file']}")
        if actual["sample_rate"] != source_audio["sample_rate"] or actual["channels"] != 2:
            failures.append(f"audio layout mismatch: {row['output_file']}")
        if actual["codec"] != "flac":
            failures.append(f"not FLAC: {row['output_file']}")
    playlist_path = output / manifest["playlist"]
    if not playlist_path.exists():
        failures.append("playlist missing")
        entries = []
    else:
        entries = [
            line.lstrip("\ufeff")
            for line in playlist_path.read_text(encoding="utf-8-sig").splitlines()
            if line and not line.startswith("#")
        ]
        expected = [row["output_file"] for row in tracks]
        if entries != expected:
            failures.append("playlist order/content differs from manifest")
    if not (output / manifest["html_player"]).exists():
        failures.append("HTML player missing")
    report = {
        "schema": SCHEMA,
        "verified_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "source_files_preserved": all((source_dir / row["source_file"]).exists() for row in tracks),
        "track_count": len(tracks),
        "playlist_entries": len(entries),
        "failures": failures,
        "pass": not failures,
    }
    (output / "verification-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    """Dispatch the separate, package, verify, or complete album stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["separate", "package", "verify", "all"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, help="Python environment containing Demucs")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    parser.add_argument("--force", action="store_true", help="Re-encode existing packaged FLAC files")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"separate", "all"}:
        separate(config, args.python, force=args.force)
    if args.stage in {"package", "all"}:
        package(config, args.ffmpeg, args.ffprobe, force=args.force)
    if args.stage in {"verify", "all"}:
        return 0 if verify(config, args.ffprobe)["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
