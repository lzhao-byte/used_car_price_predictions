import numpy as np
import pytest

from tests.conftest import ModelBuilder, make_listings


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # train() sleeps between steps so the Streamlit UI can show progress
    monkeypatch.setattr("utils.model_trains.time.sleep", lambda *_: None)


@pytest.fixture
def listings():
    # drop the string/geo columns the Model Training page never sees at this point
    return make_listings(n=1400).drop("posting_date")


def run_training(df, **kwargs):
    mb = ModelBuilder(df)
    messages = list(mb.train(**kwargs))
    return mb, messages


class TestSplit:
    def test_last_1000_rows_are_held_out_for_the_simulator(self, listings):
        mb = ModelBuilder(listings)
        x_train, x_test, y_train, y_test = mb._split_data(test_size=0.25)
        assert len(mb.sim) == 1000
        assert len(x_train) + len(x_test) == listings.height - 1000
        # the simulator rows must not leak into train/test
        assert set(mb.sim.index).isdisjoint(x_train.index)
        assert set(mb.sim.index).isdisjoint(x_test.index)

    def test_target_is_not_in_the_features(self, listings):
        mb = ModelBuilder(listings)
        x_train, *_ = mb._split_data()
        assert "price" not in x_train.columns

    def test_split_is_reproducible(self, listings):
        a = ModelBuilder(listings)._split_data()
        b = ModelBuilder(listings)._split_data()
        assert np.array_equal(a[2], b[2])


class TestEncoders:
    @pytest.mark.parametrize("option", ["ordinal", "one-hot", "target"])
    def test_supported_encoders(self, listings, option):
        mb = ModelBuilder(listings)
        mb._setup_encoder(train_opt=option)
        assert mb.encoder is not None

    def test_ordinal_encoder_maps_unseen_categories_to_minus_one(self, listings):
        mb = ModelBuilder(listings)
        mb._setup_encoder("ordinal")
        mb.encoder.fit([["a"], ["b"]])
        assert mb.encoder.transform([["zzz"]])[0][0] == -1


class TestTraining:
    @pytest.mark.parametrize("algo", ["random_forest", "linear", "decision_tree", "xgboost"])
    def test_models_learn_the_synthetic_price_signal(self, listings, algo):
        mb, messages = run_training(
            listings, sel_model=algo, sel_train_opt="ordinal", test_size=0.2
        )
        assert any("Cross Validation Score" in m for m in messages), messages
        assert not any(m.startswith("Error") for m in messages), messages
        r2 = float(mb.evaluate()["R Squared"])
        assert r2 > 0.7, f"{algo} R^2 was {r2}"

    def test_progress_messages_come_in_pipeline_order(self, listings):
        _, messages = run_training(listings, sel_train_opt="one-hot")
        assert "encoder" in messages[0]
        assert "scaler" in messages[1]
        assert "Split" in messages[2]
        assert messages[-1].startswith("Cross Validation Score")

    def test_predict_on_new_data_ignores_the_target_column(self, listings):
        mb, _ = run_training(listings, sel_model="linear", sel_train_opt="ordinal")
        preds = mb.predict(mb.sim)
        assert len(preds) == len(mb.sim)
        assert np.isfinite(preds).all()

    def test_tree_family_assignment(self, listings):
        mb, _ = run_training(listings, sel_model="xgboost", sel_train_opt="ordinal")
        assert mb.family == "tree"
        mb, _ = run_training(listings, sel_model="linear", sel_train_opt="ordinal")
        assert mb.family == "linear"


class TestEvaluate:
    def test_metrics_for_known_predictions(self, listings):
        mb = ModelBuilder(listings)
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 330.0])
        out = mb.evaluate(y_true=y_true, y_pred=y_pred)
        assert out["Mean Absolute Error"] == "16.67"  # (10 + 10 + 30) / 3
        assert out["Root Mean Squared Error"] == "19.15"  # sqrt((100 + 100 + 900) / 3)
        assert float(out["R Squared"]) == pytest.approx(0.945, abs=0.006)  # 1 - 1100 / 20000

    def test_perfect_predictions(self, listings):
        mb = ModelBuilder(listings)
        y = np.array([1.0, 2.0, 3.0])
        out = mb.evaluate(y_true=y, y_pred=y)
        assert out["Mean Absolute Error"] == "0.00"
        assert out["R Squared"] == "1.00"
