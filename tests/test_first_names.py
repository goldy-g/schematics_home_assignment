"""The small first-name table answers the original gender feature like the names database does (names made up).
The full proof, over every name in the database and the data, is scripts/names_table_check.py."""
import gzip
import json
import sys

from pipeline.ranker import FirstNameGender

TABLE = {"Dana": {"Male": 0.2, "Female": 0.8}, "Noam": {"Male": 0.5, "Female": 0.5}, "Ziv": {"Male": 1.0}}


def small(tmp_path):
    path = tmp_path / "first_name_gender.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(TABLE, f)
    return FirstNameGender(path)


def test_search_matches_names_dataset_shape(tmp_path):
    names = small(tmp_path)
    assert names.search("  dana ") == {"first_name": {"gender": {"Male": 0.2, "Female": 0.8}}, "last_name": None}
    assert names.search("Nobody") == {"first_name": None, "last_name": None}


def test_their_gender_function_reads_it(tmp_path):
    names = small(tmp_path)
    saved = sys.modules.get("bl_exp_payout_predictor")
    original = names.import_predictor()
    try:
        detect = original.BLPayoutModelsPredict.detect_gender_with_confidence
        assert detect(None, "Dana") == ("female", 0.8)
        assert detect(None, "Noam") == ("male", 0.5)  # a tie goes to "male", as in the original code
        assert detect(None, "Ziv") == ("male", 1.0)
        assert detect(None, "Nobody") == ("unknown", 0.0)
        stand_in_left = "names_dataset" in sys.modules and not hasattr(sys.modules["names_dataset"], "__file__")
        assert not stand_in_left  # the real package (or nothing) is back in place
    finally:
        if saved is None:
            sys.modules.pop("bl_exp_payout_predictor", None)  # leave no stand-in behind for other tests
