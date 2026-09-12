import schema


def test_case_id_from_path_uses_parent_folder_name():
    assert schema.case_id_from_path("data/subject001/orig1.nii") == "subject001"
    assert schema.case_id_from_path("/abs/path/subject007/orig7.nii.gz") == "subject007"


def test_case_id_from_path_falls_back_to_filename_with_no_parent_dir():
    assert schema.case_id_from_path("orig1.nii.gz") == "orig1"
    assert schema.case_id_from_path("orig1.nii") == "orig1"
