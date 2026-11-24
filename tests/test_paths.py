import pytest

from karaokeizer.paths import safe_component


def test_safe_component_accepts_normal_album_name():
    assert safe_component("Album Name (Deluxe)", field="album") == "Album Name (Deluxe)"


@pytest.mark.parametrize("value", ["..", "../escape", "folder/name", "bad:name", "trailing.", "bad\x00name"])
def test_safe_component_rejects_traversal_and_nonportable_names(value):
    with pytest.raises(ValueError):
        safe_component(value, field="album")
