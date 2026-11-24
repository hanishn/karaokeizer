from karaokeizer.build_karaoke_overlay import (
    filter_lyric_segments,
    note_for_midi,
    segment_timing_errors,
)


def word(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end, "probability": 0.95}


def test_note_names_include_pitch_class_and_octave():
    assert note_for_midi(60) == "C4"
    assert note_for_midi(69) == "A4"
    assert note_for_midi(None) is None


def test_segment_gate_accepts_monotonic_positive_timing():
    segment = {"start": 1.0, "end": 2.0, "words": [word("hello", 1.1, 1.5), word("world", 1.5, 1.9)]}
    assert segment_timing_errors(segment) == []


def test_segment_gate_rejects_collapsed_duplicate_words():
    segment = {"start": 1.0, "end": 2.0, "words": [word("hello", 1.1, 1.1), word("world", 1.1, 1.8)]}
    failures = segment_timing_errors(segment)
    assert "collapsed_word" in failures
    assert "duplicate_word_start" in failures


def test_filter_never_renders_malformed_segments():
    valid = {"start": 1.0, "end": 2.0, "text": "hello", "words": [word("hello", 1.1, 1.8)]}
    invalid = {"start": 3.0, "end": 4.0, "text": "bad", "words": [word("bad", 3.2, 3.2)]}
    accepted, status = filter_lyric_segments([valid, invalid], {"lyric_classification": {}})
    assert accepted == [valid]
    assert status == "needs_manual_review"


def test_explicit_instrumental_classification_suppresses_all_lyrics():
    segment = {"start": 1.0, "end": 2.0, "text": "hallucination", "words": [word("hallucination", 1.1, 1.8)]}
    accepted, status = filter_lyric_segments(
        [segment], {"lyric_classification": {"status": "instrumental_or_no_reliable_vocal"}}
    )
    assert accepted == []
    assert "instrumental" in status.lower() or "reliable" in status.lower()
