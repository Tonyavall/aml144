from src.data.images import list_all_test_images


def test_lists_all_test_images_in_numeric_order(tmp_path):
    # filenames must sort numerically by stem, not lexicographically
    for stem in [0, 1, 2, 10, 100, 999, 1035]:
        (tmp_path / f"{stem}.jpg").write_bytes(b"x")

    ids, paths = list_all_test_images(tmp_path)

    assert ids == [
        "0.jpg",
        "1.jpg",
        "2.jpg",
        "10.jpg",
        "100.jpg",
        "999.jpg",
        "1035.jpg",
    ]

    assert len(paths) == len(ids) == 7
    assert [p.name for p in paths] == ids
