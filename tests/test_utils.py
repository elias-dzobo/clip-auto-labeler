from pathlib import Path

from clip_auto_labeler import utils


def test_chunked_splits_sequences_evenly():
    data = [1, 2, 3, 4, 5]
    chunks = list(utils.chunked(data, 2))
    assert chunks == [[1, 2], [3, 4], [5]]


def test_read_labels_file(tmp_path: Path):
    file_path = tmp_path / "labels.txt"
    file_path.write_text("cat\n\ndog\n", encoding="utf-8")
    assert utils.read_labels_file(file_path) == ["cat", "dog"]

