import pandas as pd


def build_submission_df(id_to_label, required_ids):
    rows = [{"ID": rid, "Label": id_to_label[rid]} for rid in required_ids]
    return pd.DataFrame(rows, columns=["ID", "Label"])


def validate_submission(df, required_ids, valid_labels):
    assert list(df.columns) == ["ID", "Label"], "columns must be exactly ID,Label"
    assert df["ID"].tolist() == list(required_ids), "ids must match the expected order"
    assert len(df) == len(required_ids), "row count must match the expected set"
    valid = {str(v) for v in valid_labels}
    assert set(df["Label"].astype(str)).issubset(valid), (
        "labels must be in the valid set"
    )


def write_submission(id_to_label, required_ids, out_path, valid_labels):
    df = build_submission_df(id_to_label, required_ids)
    validate_submission(df, required_ids, valid_labels)
    df.to_csv(out_path, index=False)
    return df
