import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DataFrame
from shap import Explanation
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xarray import Dataset

import xrml as xl
from xrml.utils import Model, make_pandas_pipeline, precision_at_k, recall_at_k


def _data() -> DataFrame:
    data = load_breast_cancer()
    return (
        pd.DataFrame(data["data"], columns=data["feature_names"])
        .assign(target=data["target"])
        .assign(row_id=lambda df: np.arange(len(df)))
        .assign(row_id=lambda df: "I" + df["row_id"].astype(str))
        .assign(filename=data["filename"])
        .assign(descr=data["DESCR"])
    )


@pytest.fixture(scope="module")
def data() -> DataFrame:
    return _data()


def _dataset(df: DataFrame) -> Dataset:
    return xl.create_dataset(df, index=["row_id"], outcomes=["target"])


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return _dataset(_data())


def estimator_fn_2_ests() -> list[Model]:
    return [
        Model(
            "lr",
            "Logistic Regression",
            make_pandas_pipeline(
                StandardScaler(),
                SelectKBest(k=3, score_func=f_classif),
                LogisticRegression(random_state=0),
            ),
        ),
        Model(
            "gbm",
            "Gradient Boosting Classifier",
            make_pandas_pipeline(GradientBoostingClassifier(random_state=0)),
        ),
    ]


def _prep(dataset: Dataset, data: DataFrame) -> Dataset:
    oos_df = (
        data.sample(n=3, random_state=0)
        .assign(target=np.nan)
        .assign(row_id=lambda df: "O" + df["row_id"])
    )
    return (
        dataset.pipe(xl.add_stratified_splits, n_splits=5)
        .pipe(xl.add_models, estimator_fn=estimator_fn_2_ests)
        .pipe(
            xl.add_unlabeled_data,
            df=oos_df,
            index=["row_id"],
            fold_label=np.nan,
        )
    )


def test_create_dataset(data: DataFrame) -> None:
    ds = _dataset(data)
    assert dict(ds.dims) == {
        "descriptors": 2,
        "features": 30,
        "index": 569,
        "outcomes": 1,
    }
    assert set(ds.features.values) == set(
        data.drop(columns=["row_id", "filename", "descr", "target"]).columns
    )
    assert set(ds.descriptors.values) == {"descr", "filename"}
    assert set(ds.outcomes.values) == {"target"}


def test_create_splits(dataset: Dataset) -> None:
    ds = xl.add_stratified_splits(dataset, n_splits=5)
    assert ds.dims["folds"] == 5
    assert set(ds.fold.dims) == {"index", "folds"}
    assert set(ds.folds.values) == {0, 1, 2, 3, 4}
    assert ds.fold.to_series().value_counts().to_dict() == {"train": 2276, "test": 569}


def test_create_splits_groups(dataset: Dataset) -> None:
    dataset = dataset.assign(
        high_symmetry_err=lambda ds: ds.sel(features="symmetry error", drop=True)[
            "feature"
        ]
        > 0.03
    )
    ds = xl.add_stratified_splits(
        dataset,
        n_splits=5,
        groups="high_symmetry_err",
    )
    assert ds.dims["folds"] == 5
    assert set(ds.fold.dims) == {"index", "folds"}
    assert set(ds.folds.values) == {0, 1, 2, 3, 4}
    assert ds.fold.to_series().value_counts().to_dict() == {"train": 2276, "test": 569}
    # check that binary symmetry error grp is only in one fold
    sym_err = ds.sel(features="symmetry error", drop=True)["feature"]
    assert (
        np.asarray(
            [
                (sym_err[(ds.sel(folds=f)["fold"] == "test")] > 0.03).sum()
                for f in ds["folds"].values
            ]
        )
        == 0
    ).sum() == 4


def test_add_models(dataset: Dataset, data: DataFrame) -> None:
    ds = _prep(dataset, data)
    assert ds.dims["estimators"] == 2
    assert set(ds.estimator.dims) == {"estimators", "folds"}
    assert set(ds.estimators.values) == {"lr", "gbm"}


@pytest.mark.parametrize("shap_mode", ["true-to-model", "true-to-data", None])
@pytest.mark.parametrize(
    ("model", "method", "explainer_kwargs"),
    [
        ("gbm", "tree-shap", {"model_output": "raw"}),
        ("lr", "linear-shap", {}),
        ("lr", "linear-hamard", {}),
    ],
)
def test_run_explain(
    shap_mode: Any,
    model: str,
    method: Any,
    explainer_kwargs: dict[str, Any],
    dataset: Dataset,
    data: DataFrame,
) -> None:
    ds = xl.explain(
        xl.run_classification_cv(_prep(dataset, data)),
        model=model,
        shap_mode=shap_mode,
        method=method,
        tree_shap_check_additivity=False,
        **explainer_kwargs,
    )
    assert ds.shap_value.dims == ("models", "index", "features")
    assert ds.shap_feature.dims == ("models", "index", "features")
    assert ds.shap_base_value.dims == ("models", "index")

    # Select data from a single fold
    dsf = ds.sel(models=model, folds=0, index=ds.fold.sel(folds=0).values == "test")
    feature_names = dsf.features.values.tolist()
    pipeline = dsf.model.item(0).estimator
    # Validate that features dropped do not affect explanations, if there is
    # transformation pipeline preceding the estimator
    if len(pipeline.steps) > 1:
        selected_feature_names = pipeline[:-1].get_feature_names_out(feature_names)
        ignored_feature_names = list(set(feature_names) - set(selected_feature_names))
        assert len(set(selected_feature_names) - set(feature_names)) == 0
        assert (
            dsf.shap_value.sel(features=selected_feature_names).notnull().all().item(0)
        )
        if ignored_feature_names:
            assert (
                dsf.shap_value.sel(features=ignored_feature_names)
                .isnull()
                .all()
                .item(0)
            )
    else:
        assert dsf.shap_value.sel(features=feature_names).notnull().all().item(0)


def test_run_explain_twice_corner_broadcast(dataset: Dataset, data: DataFrame) -> None:
    ds = xl.explain(
        xl.run_classification_cv(_prep(dataset, data)),
        model="gbm",
        shap_mode="true-to-data",
        score_name_max="roc_auc",
        method="tree-shap",
    )
    gbm_ds = ds.sel(models="gbm")
    e = xl.shap_explanation(gbm_ds)
    assert isinstance(e, Explanation)
    assert np.all(e.feature_names == gbm_ds["features"].values)
    assert np.allclose(e.data, gbm_ds["shap_feature"].values)
    assert np.allclose(e.values, gbm_ds["shap_value"].values)
    assert np.allclose(e.base_values, gbm_ds["shap_base_value"].values)
    assert np.all(e.display_data == gbm_ds["index"].values)


def test_run_cv(dataset: Dataset, data: DataFrame) -> None:
    ds = xl.run_classification_cv(_prep(dataset, data))
    assert ds.dims["models"] == 2
    assert set(ds.model.dims) == {"models", "folds"}
    assert set(ds.models.values) == {"lr", "gbm"}
    assert ds.classes.to_series().to_list() == ["negative", "positive"]
    assert set(ds.prediction.dims) == {"index", "classes", "models", "outcomes"}

    # Check that models perform as expected on this data
    assert (ds.score.to_series().unstack()["average_precision"] > 0.95).all()
    assert (ds.score.to_series().unstack()["roc_auc"] > 0.95).all()

    # Ensure that the models fit were never accidentally based
    # on shared references across folds
    for i1, m1 in enumerate(ds.model.values):
        for i2, m2 in enumerate(ds.model.values):
            if i1 != i2:
                assert m1 is not m2


def test_run_cv_reproducibility() -> None:
    # Ensure that results are identical across runs by default
    ds1 = xl.run_classification_cv(_prep(_dataset(_data()), _data()))
    ds2 = xl.run_classification_cv(_prep(_dataset(_data()), _data()))

    # Check that fitted models are equivalent
    # TODO: Is there a better way to compare scikit-learn estimators?
    pd.testing.assert_series_equal(
        ds1.model.to_series().apply(pickle.dumps),
        ds2.model.to_series().apply(pickle.dumps),
    )
    # Check that performance scores are equivalent
    xr.testing.assert_equal(ds1.score, ds2.score)


@pytest.mark.parametrize(
    "y_true,y_score,k,expected",
    [
        ([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5], 2, 1.0),
        ([0, 0, 0, 1, 1], [0.1, 0.2, 0.4, 0.3, 0.5], 2, 1 / 2.0),
        ([0, 0, 0, 1, 1], [0.1, 0.4, 0.5, 0.2, 0.3], 2, 0.0),
        ([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5], 3, 2 / 3.0),
        ([0, 0, 0, 1, 1], [0.1, 0.4, 0.5, 0.2, 0.3], 10, 0.4),
    ],
)
def test_precision_at_k(
    y_true: list[int], y_score: list[float], k: int, expected: float
) -> None:
    assert expected == precision_at_k(np.array(y_true), np.array(y_score), k)


@pytest.mark.parametrize(
    "y_true,y_score,k,expected",
    [
        ([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5], 2, 1.0),
        ([0, 0, 0, 1, 1], [0.1, 0.2, 0.4, 0.3, 0.5], 2, 1 / 2.0),
        ([0, 0, 0, 1, 1], [0.1, 0.4, 0.5, 0.2, 0.3], 2, 0.0),
        ([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5], 3, 1.0),
        ([0, 0, 0, 1, 1], [0.1, 0.4, 0.5, 0.2, 0.3], 10, 1.0),
    ],
)
def test_recall_at_k(
    y_true: list[int], y_score: list[float], k: int, expected: float
) -> None:
    assert expected == recall_at_k(np.array(y_true), np.array(y_score), k)


def test_get_classification_scores() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.0, 0.1, 0.2, 0.7, 0.6, 1.0])
    scores = xl.get_classification_scores(y_true, y_score)
    assert scores["n_samples"] == 6
    assert scores["n_positives"] == 2
    assert scores["balance"] == pytest.approx(2 / 6)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == 1.0
    assert scores["roc_auc"] == 0.875
    assert scores["average_precision"] == pytest.approx(5 / 6)
    assert scores["brier"] == pytest.approx(7 / 60)
    assert scores["balanced_mae"] == 0.225
    assert scores["log_loss"] == pytest.approx(0.3405504158439941)
    assert sorted(xl.get_classification_score_names()) == sorted(scores.keys())


def test_get_regression_scores() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.0, 0.1, 0.2, 0.7, 0.6, 1.0])
    scores = xl.get_regression_scores(y_true, y_score)
    assert scores["n_samples"] == 6
    assert scores["explained_variance"] == pytest.approx(0.52)
    assert scores["mean_squared_error"] == pytest.approx(0.11666666666666665)
    assert scores["mean_absolute_error"] == pytest.approx(0.2333333333333333)
    assert scores["r2"] == pytest.approx(0.4750000000000001)
    assert sorted(xl.get_regression_score_names()) == sorted(scores.keys())


def test_create_dataset_no_desc() -> None:
    ds = xl.create_dataset(
        _data().drop(columns=["descr", "filename"]),
        index=["row_id"],
        outcomes=["target"],
    )
    assert "descriptors" not in ds
    assert "descriptor" not in ds


def test_fit_multiple_estimators_single_split(dataset: Dataset) -> None:
    dataset_fit = (
        dataset.pipe(xl.add_single_split)
        .pipe(xl.add_models, estimator_fn_2_ests)
        .pipe(xl.fit)
        .pipe(xl.predict)
    )
    assert {"models", "model", "prediction"}.issubset(set(dataset_fit.variables.keys()))
    assert set(dataset_fit["models"].values) == {"lr", "gbm"}
