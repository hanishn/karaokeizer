import pytest

from karaokeizer.build_karaoke_overlay import analysis_cache_key, contour_to_note_events, validate_analysis_overrides


def test_analysis_cache_key_changes_with_every_material_input():
    track = {"vocal_sha256": "v1", "accompaniment_sha256": "a1"}
    first = analysis_cache_key(track, "t1")
    assert first != analysis_cache_key({**track, "vocal_sha256": "v2"}, "t1")
    assert first != analysis_cache_key({**track, "accompaniment_sha256": "a2"}, "t1")
    assert first != analysis_cache_key(track, "t2")


def test_same_pitch_contour_is_split_at_word_boundaries():
    contour = [
        {"time": time, "midi": 60.1, "probability": 0.9}
        for time in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    ]
    words = [
        {"word_index": 0, "text": "hold", "start": 0.0, "end": 0.4},
        {"word_index": 1, "text": "on", "start": 0.4, "end": 0.9},
    ]
    events = contour_to_note_events(contour, words)
    assert {event["word_index"] for event in events} == {0, 1}
    assert all(event["lyric"] in {"hold", "on"} for event in events)


def test_brief_vibrato_excursion_does_not_fragment_note_event():
    contour = [
        {"time": index * 0.05, "midi": midi, "probability": 0.95}
        for index, midi in enumerate([60, 60, 60, 61, 61, 60, 60, 60, 60])
    ]
    words = [{"word_index": 0, "text": "held", "start": 0.0, "end": 0.5}]

    events = contour_to_note_events(contour, words)
    assert len(events) == 1


def test_sustained_pitch_change_creates_second_note_event():
    contour = [
        {"time": index * 0.06, "midi": midi, "probability": 0.95}
        for index, midi in enumerate([60, 60, 60, 60, 62, 62, 62, 62, 62])
    ]
    words = [{"word_index": 0, "text": "melisma", "start": 0.0, "end": 0.6}]

    events = contour_to_note_events(contour, words)
    assert [event["midi"] for event in events] == [60, 62]


def valid_review() -> dict:
    return {
        "schema": "karaokeizer-analysis-overrides/v1", "pass": True,
        "reviewer": "Reviewer", "method": "listening and score", "reviewed_utc": "2026-01-01T00:00:00Z",
        "tracks": {
            "1": {
                "status": "pass",
                "key": {"tonic": "C", "mode": "major", "evidence": "Checked against score."},
                "meter": {"numerator": 4, "denominator": 4, "downbeat_phase": 0, "evidence": "Checked measures."},
            }
        },
    }


def test_analysis_review_requires_exact_track_inventory():
    review = valid_review()
    review["tracks"] = {}
    with pytest.raises(ValueError, match="exactly match"):
        validate_analysis_overrides(review, [{"number": 1}])


def test_analysis_review_requires_per_track_evidence():
    review = valid_review()
    review["tracks"]["1"]["key"]["evidence"] = ""
    with pytest.raises(ValueError, match="incomplete"):
        validate_analysis_overrides(review, [{"number": 1}])
