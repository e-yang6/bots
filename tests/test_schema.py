import json

import pytest

import schema


def test_case_id_from_path_uses_parent_folder_name():
    assert schema.case_id_from_path("data/subject001/orig1.nii") == "subject001"
    assert schema.case_id_from_path("/abs/path/subject007/orig7.nii.gz") == "subject007"


def test_case_id_from_path_falls_back_to_filename_with_no_parent_dir():
    assert schema.case_id_from_path("orig1.nii.gz") == "orig1"
    assert schema.case_id_from_path("orig1.nii") == "orig1"


def prediction():
    return schema.make_prediction("subject001", [schema.make_daughter(
        "branch_001", [0, 0, 0], [5, 0, 0], 1.2, [1, 0, 0]
    )])


def test_submission_strips_diagnostic_fields(tmp_path):
    data = prediction()
    data["daughters"][0]["confidence"] = 0.9
    output = tmp_path / "prediction.json"
    schema.write_prediction(data, output)
    written = json.loads(output.read_text())
    assert set(written) == {"case_id", "parent", "daughters"}
    assert set(written["daughters"][0]) == {
        "instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz"
    }
    assert data["daughters"][0]["confidence"] == 0.9


@pytest.mark.parametrize("field,value", [
    ("radius_mm", float("nan")), ("radius_mm", -1), ("radius_mm", True),
    ("ostium_xyz_mm", [0, float("inf"), 0]), ("seed_xyz_mm", [1, 2]),
    ("direction_xyz", [0, 0, 0]), ("direction_xyz", [2, 0, 0]),
    ("direction_xyz", [-1, 0, 0]), ("parent_instance_id", "other"),
    ("instance_id", "renal"),
])
def test_invalid_prediction_is_rejected_before_writing(tmp_path, field, value):
    data = prediction()
    data["daughters"][0][field] = value
    output = tmp_path / "prediction.json"
    with pytest.raises(ValueError):
        schema.write_prediction(data, output)
    assert not output.exists()


def test_duplicate_ids_are_rejected(tmp_path):
    data = prediction()
    data["daughters"].append(dict(data["daughters"][0]))
    with pytest.raises(ValueError):
        schema.write_prediction(data, tmp_path / "prediction.json")


def test_empty_submission_roundtrip(tmp_path):
    data = schema.make_prediction("subject001")
    output = tmp_path / "prediction.json"
    schema.write_prediction(data, output)
    assert schema.read_prediction(output) == data
