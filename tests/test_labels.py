from src.data.labels import build_class_to_idx, idx_to_class


def test_integer_folders_are_sorted_numerically(tmp_path):
    # the imagefolder trap: string sort would map "10" -> index 2
    for name in ["0", "1", "2", "10", "11", "99"]:
        (tmp_path / name).mkdir()

    mapping = build_class_to_idx(tmp_path)

    assert mapping["0"] == 0
    assert mapping["2"] == 2
    assert mapping["10"] == 10
    assert mapping["11"] == 11
    assert mapping["99"] == 99


def test_idx_to_class_round_trips(tmp_path):
    for name in ["0", "5", "10"]:
        (tmp_path / name).mkdir()

    mapping = build_class_to_idx(tmp_path)

    inv = idx_to_class(mapping)

    assert inv[mapping["10"]] == "10"
    assert inv[10] == "10"


def test_non_integer_folders_fall_back_to_string_sort(tmp_path):
    for name in ["cat", "dog", "ant"]:
        (tmp_path / name).mkdir()

    mapping = build_class_to_idx(tmp_path)

    assert mapping == {"ant": 0, "cat": 1, "dog": 2}


def test_hidden_directories_are_ignored(tmp_path):
    # stray tooling dirs like .pytest_cache must not be treated as classes
    for name in ["0", "1", "2"]:
        (tmp_path / name).mkdir()

    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".ipynb_checkpoints").mkdir()

    mapping = build_class_to_idx(tmp_path)

    assert mapping == {"0": 0, "1": 1, "2": 2}
