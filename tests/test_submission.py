import pandas as pd
import pytest

from src.submission import build_submission_df, validate_submission, write_submission

VALID = [str(i) for i in range(100)]


def test_writer_matches_required_ids_exactly(tmp_path):
    required = ["0.jpg", "1.jpg", "2.jpg"]
    id_to_label = {"0.jpg": "5", "1.jpg": "10", "2.jpg": "99"}
    out = tmp_path / "submission.csv"

    write_submission(id_to_label, required, out, VALID)

    written = pd.read_csv(out)

    assert list(written.columns) == ["ID", "Label"]
    assert written["ID"].tolist() == ["0.jpg", "1.jpg", "2.jpg"]
    assert written["Label"].tolist() == [5, 10, 99]
    assert len(written) == 3


def test_validate_rejects_out_of_range_label():
    df = build_submission_df({"0.jpg": "200"}, ["0.jpg"])

    with pytest.raises(AssertionError):
        validate_submission(df, ["0.jpg"], VALID)


def test_validate_rejects_id_mismatch():
    df = build_submission_df({"0.jpg": "5"}, ["0.jpg"])

    with pytest.raises(AssertionError):
        validate_submission(df, ["0.jpg", "1.jpg"], VALID)
