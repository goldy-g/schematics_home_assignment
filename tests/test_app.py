"""The HTTP layer, with a stand-in ranker (no models, no network)."""
from fastapi.testclient import TestClient

from pipeline import settings
from serving import app as api


class StandInRanker:
    def rank(self, visitor):
        return {"xlt": {"rank": 1.0, "expected_payout": 9.5}}, "tabpfn"


class PlainPyfuncModel:
    """A model with only MLflow's standard predict(), no rank()."""
    def predict(self, frame):
        assert isinstance(frame.loc[0, "sub1"], str)  # numbers are turned into text, as their code would
        return {"xlt": {"rank": 1.0, "expected_payout": 3.0}}


def client(ranker=StandInRanker(), model=None):
    api.state.update(model=model, ranker=ranker, version="7", run_id="run", loaded_at="now")
    return TestClient(api.app)  # used without "with", so the background champion watcher does not start


def test_rank_answers_in_their_format_with_headers():
    response = client().post("/rank", json=settings.EXAMPLE_VISITOR)
    assert response.status_code == 200
    assert response.json() == {"xlt": {"rank": 1.0, "expected_payout": 9.5}}
    assert response.headers["X-Model-Version"] == "7"
    assert response.headers["X-Payout-Source"] == "tabpfn"


def test_their_example_with_numbers_is_accepted():
    their_style = dict(settings.EXAMPLE_VISITOR, sub1=1121993, sub3=1513124082)
    assert client().post("/rank", json=their_style).status_code == 200


def test_missing_register_date_is_rejected():
    visitor = dict(settings.EXAMPLE_VISITOR)
    del visitor["register_date"]
    response = client().post("/rank", json=visitor)
    assert response.status_code == 422
    assert "register_date" in response.json()["detail"]


def test_missing_required_field_is_rejected():
    visitor = dict(settings.EXAMPLE_VISITOR)
    del visitor["page"]
    assert client().post("/rank", json=visitor).status_code == 422


def test_not_ready_until_a_model_is_loaded():
    api.state.update(ranker=None, version=None)
    test_client = TestClient(api.app)
    assert test_client.get("/health").status_code == 503
    assert test_client.post("/rank", json=settings.EXAMPLE_VISITOR).status_code == 503


def test_a_model_without_rank_still_answers():
    plain = PlainPyfuncModel()
    response = client(ranker=plain, model=plain).post("/rank", json=dict(settings.EXAMPLE_VISITOR, sub1=1121993))
    assert response.status_code == 200
    assert response.json() == {"xlt": {"rank": 1.0, "expected_payout": 3.0}}
    assert response.headers["X-Payout-Source"] == "unknown"
