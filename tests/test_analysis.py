from karaokeizer.build_karaoke_overlay import analysis_cache_key, contour_to_note_events


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
