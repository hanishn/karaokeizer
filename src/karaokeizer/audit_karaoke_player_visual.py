"""Audit rendered karaoke layout across every lyric-bearing track using Playwright."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


def safe_name(value: str) -> str:
    """Convert a label into a stable screenshot filename component."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def sha256(path: Path) -> str:
    """Hash the exact player document exercised by the browser audit."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    """Exercise representative player times in Chromium and retain evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--player", required=True, type=Path, help="Local player file bound into the audit evidence")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--samples-per-track", type=int, default=3)
    parser.add_argument(
        "--sample", action="append", default=[], metavar="TRACK:SECONDS",
        help="Add a known-risk sample (repeatable), for example --sample 5:166",
    )
    args = parser.parse_args()
    extra_samples: dict[int, list[float]] = {}
    for value in args.sample:
        track, seconds = value.split(":", 1)
        extra_samples.setdefault(int(track), []).append(float(seconds))
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        inventory = page.evaluate("tracks.map((t,index)=>({index,title:t.title,segments:t.segments.map(s=>({start:s.start,end:s.end}))}))")
        for track in inventory:
            segments = track["segments"]
            if not segments:
                sample_times = [30.0]
            else:
                positions = [round(i * (len(segments) - 1) / max(1, args.samples_per_track - 1)) for i in range(args.samples_per_track)]
                sample_times = [round((segments[index]["start"] + segments[index]["end"]) / 2, 3) for index in sorted(set(positions))]
            sample_times = sorted(set(sample_times + extra_samples.get(track["index"] + 1, [])))
            samples = []
            for sample_index, time in enumerate(sample_times, 1):
                page.locator("#list button").nth(track["index"]).evaluate("button=>button.click()")
                result = page.evaluate(
                    """time => {
                        audio.currentTime=time; render();
                        const t=tracks[current],v=notationWindow(t,time),lanes=stableWordLaneMap(t),events=visualNoteEvents(t);
                        sctx.save();sctx.font='800 19px system-ui';
                        const unboundWords=[],clippedWords=[];
                        const visible=t.words.map(w=>{
                            const notes=events.filter(n=>n.word_index===w.word_index);
                            const start=(notes[0]?.start??w.start)+lyricBias,end=Math.max(effectiveEnd(w),...notes.map(n=>n.end))+lyricBias;
                            if(end<v.start||start>v.end)return null;
                            if(!notes.length)unboundWords.push(w.text);
                            const plotStart=drawTimeSignature(t,drawKeySignature(t)+10),x=plotStart+(start-v.start)/(v.end-v.start)*(staff.width-plotStart),width=sctx.measureText(String(w.text||'').trim()).width;
                            if(x<plotStart+8||x+width>staff.width-8){clippedWords.push(w.text);return null}
                            return {text:w.text,x,width,lane:lanes.get(w),start,end};
                        }).filter(Boolean);sctx.restore();
                        const collisions=[];
                        for(let i=0;i<visible.length;i++)for(let j=i+1;j<visible.length;j++)if(visible[i].lane===visible[j].lane&&visible[i].x<visible[j].x+visible[j].width&&visible[j].x<visible[i].x+visible[i].width)collisions.push([visible[i].text,visible[j].text]);
                        return {time,state:analysisState.textContent,kind:analysisState.className,visible,collisions,unboundWords,clippedWords,horizontalOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth};
                    }""",
                    time,
                )
                screenshot = args.evidence_dir / f"{track['index'] + 1:02d}-{safe_name(track['title'])}-{sample_index}.png"
                page.locator("#staff").screenshot(path=str(screenshot))
                result["screenshot"] = str(screenshot)
                samples.append(result)
            failures = []
            if any(sample["collisions"] for sample in samples):
                failures.append("word_collision")
            if any(sample["horizontalOverflow"] for sample in samples):
                failures.append("horizontal_page_overflow")
            if any(sample["unboundWords"] for sample in samples):
                failures.append("visible_lyric_without_note_binding")
            if any(sample["clippedWords"] for sample in samples):
                failures.append("lyric_clipped_at_notation_edge")
            rows.append({"track": track["index"] + 1, "title": track["title"], "samples": samples, "failures": failures, "pass": not failures})
        browser.close()
    report = {
        "schema": "karaoke-player-visual-audit/v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "url": args.url,
        "player": str(args.player),
        "player_sha256": sha256(args.player),
        "tracks": rows,
        "page_errors": page_errors,
        "passing_tracks": sum(row["pass"] for row in rows),
        "track_count": len(rows),
        "pass": not page_errors and all(row["pass"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passing_tracks": report["passing_tracks"], "track_count": report["track_count"], "page_errors": len(page_errors), "pass": report["pass"]}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
