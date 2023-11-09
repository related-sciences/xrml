import inspect
import json
import logging
import re
import warnings
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from functools import wraps
from keyword import iskeyword
from pathlib import Path
from typing import Any, Literal

import dill
import fsspec
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
import tqdm
import xarray as xr
from fsspec import AbstractFileSystem, get_filesystem_class
from fsspec.utils import get_protocol
from pandas import DataFrame
from shap import Explanation
from shap.maskers import Independent
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.multiclass import type_of_target
from xarray import DataArray, Dataset

logger = logging.getLogger(__name__)


@dataclass
class Model:
    slug: str
    """ Human-readable unique model identifier, e.g. 'logreg' """

    name: str
    """ Display name, e.g. 'Logistitc Regression' """

    estimator: BaseEstimator
    """ Estimator instance """

    properties: dict[str, Any] = dataclass_field(default_factory=lambda: {})
    """ Arbitrary properties/metadata for the model """


def sanitize_column_name(name: str, strict: bool = True) -> str:
    """
    This method sanitizes a string for use in URL, path, column names etc.
    Lowercases, replaces non-alpha-numeric characters with `_`,
    removes multiple consecutive `_`, and rstrips `_`.
    """
    n = re.sub(r"[^0-9a-z]+", "_", name.lower())
    r = n.rstrip("_")
    if not r:
        raise ValueError(f"{name} does not contain any valid characters")
    # If digit at the beginning prefix with _
    if strict:
        r = r if not r[0].isdigit() else f"_{r}"
        r = r if r.isidentifier() and not iskeyword(r) else f"_{r}"
        r = r if not r[0] == "_" else f"f{r}"
    return r


def set_output(f: Any, transform_type: Literal["pandas", "default"]) -> Any:
    @wraps(f)
    def wrapper(*args: Any, **kwds: Any) -> Any:
        return f(*args, **kwds).set_output(transform=transform_type)

    return wrapper


make_pandas_pipeline = set_output(make_pipeline, "pandas")


#######################
# Xarray CV Functions #
#######################


def _to_dataframe(array: xr.DataArray) -> DataFrame:
    assert array.ndim == 2
    return pd.DataFrame(array, index=array[array.dims[0]], columns=array[array.dims[1]])


def split(ds: Dataset) -> tuple[DataArray | None, DataArray | None]:
    """Break dataset into features and outcomes for modeling"""
    X = None
    if "feature" in ds:
        if ds.dims["features"] < 1:
            raise ValueError("Must provide at least one feature")
        X = ds["feature"].transpose("index", "features")
        assert X.ndim == 2

    Y = None
    if "outcome" in ds:
        if ds.dims["outcomes"] < 1:
            raise ValueError("Must provide at least one outcome")
        Y = ds["outcome"].transpose("index", "outcomes")
        assert Y.ndim == 2

    return X, Y


def _split(ds: Dataset, squeeze_y: bool) -> tuple[DataFrame, DataFrame | pd.Series]:
    X, Y = split(ds)
    assert X is not None and Y is not None
    assert X.shape[0] == Y.shape[0]
    X, Y = _to_dataframe(X), _to_dataframe(Y)

    # Squeeze Y if desired to avoid scikit-learn warnings about
    # single-task outcomes in 2D arrays
    if squeeze_y and Y.shape[1] == 1:
        Y = Y.squeeze()

    return X, Y


def fit(
    ds: Dataset,
    groups: Hashable | None = None,
    ignore_convergence_warnings: bool = True,
    squeeze_y: bool = True,
) -> Dataset:
    """Fit estimators attached to a dataset."""
    X, Y = _split(ds, squeeze_y=squeeze_y)

    res = []
    estimators = np.ravel(ds.estimator.values).tolist()
    for estimator in estimators:
        est = estimator.estimator
        # Only pass a groups param to .fit if the estimator will support it, nothing that there
        # might be a mixture of models that do and do not support it in the same training run if,
        # for example, dummy regressors are included that don't actually need inner CV
        fit_params = {}
        if groups is not None:
            if isinstance(est, Pipeline):
                if "groups" in inspect.signature(est.steps[-1][1].fit).parameters:
                    fit_params = {
                        f"{est.steps[-1][0]}__groups": get_split_groups(
                            ds, groups
                        ).values
                    }
            else:
                if "groups" in inspect.signature(est.fit).parameters:
                    fit_params = {"groups": get_split_groups(ds, groups).values}
        with warnings.catch_warnings():
            if ignore_convergence_warnings:
                # Ignore convergence warnings noting that this will not work when
                # using subprocesses (e.g. as is common with `n_jobs`` != 1 in many estimators)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            est = est.fit(X, Y, **fit_params)
        model = replace(estimator, estimator=est)
        res.append(model)

    return ds.assign(model=("models", res)).assign(models=[est.slug for est in res])


def predict_proba(ds: Dataset) -> Dataset:
    """Predict binary class probabilities for all models in a dataset."""
    X, _ = split(ds.drop_vars("outcomes") if "outcomes" in ds else ds)
    assert X is not None
    X = _to_dataframe(X)
    res = []
    for model in ds.model.values:
        Y = model.estimator.predict_proba(X)
        if Y.ndim == 1:
            Y = np.concatenate([1 - Y, Y], axis=1)
        if Y.ndim > 2:
            raise ValueError(f"Predictions of shape {Y.shape} not supported")
        res.append(Y[..., np.newaxis, np.newaxis])
    return ds.assign(
        prediction=(
            ["index", "classes", "models", "outcomes"],
            np.concatenate(res, axis=2),
        )
    ).assign(classes=["negative", "positive"])


def predict(ds: Dataset) -> Dataset:
    """Predict class labels or continuous outcomes for all models in a dataset."""
    X, _ = split(ds.drop_vars("outcomes") if "outcomes" in ds else ds)
    assert X is not None
    X = _to_dataframe(X)
    res = []
    for model in ds.model.values:
        Y = model.estimator.predict(X)
        if Y.ndim == 1:
            Y = np.expand_dims(Y, axis=1)
        if Y.ndim > 2:
            raise ValueError(f"Predictions of shape {Y.shape} not supported")

        res.append(Y[:, np.newaxis, :])
    return ds.assign(
        prediction=(
            ["index", "models", "outcomes"],
            np.concatenate(res, axis=1),
        )
    )


def add_single_split(ds: Dataset, split: str = "test") -> Dataset:
    """Add dummy split to a dataset.

    This is useful for maintaining compatibility with functions that require
    folds to iterate over, though there are cases where this is unnecessary (e.g.
    in model refitting).
    """
    return ds.assign(
        fold=(("index", "folds"), np.full((ds.dims["index"], 1), split, dtype="O"))
    ).assign_coords(folds=np.arange(1))


def get_split_groups(ds: Dataset, groups: Hashable) -> DataArray:
    if groups in ds:
        group_values = ds[groups]
    elif groups in ds.indexes["index"].names:
        group_values = ds[groups]
    elif "descriptor" in ds and groups in ds.descriptors:
        group_values = ds.descriptor.sel(descriptors=groups)
    else:
        raise ValueError(f"Failed to find variable for groups '{groups}'")
    if group_values.ndim != 1:
        raise ValueError(f"Groups array must be 1D, found shape {group_values.shape}")
    return group_values


def add_group_splits(ds: Dataset, n_splits: int, groups: Hashable) -> Dataset:
    """Add splits/folds to a dataset based on groupings (typically identifiers)."""
    folds = np.full((ds.dims["index"], n_splits), "", dtype="O")
    for i, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(
            X=ds["outcome"].values,
            y=ds["outcome"].values,
            groups=get_split_groups(ds, groups).values,
        )
    ):
        folds[train, i] = "train"
        folds[test, i] = "test"

    return ds.assign(fold=(("index", "folds"), folds)).assign_coords(
        folds=np.arange(n_splits)
    )


def add_stratified_splits(
    ds: Dataset,
    n_splits: int,
    seed: int = 0,
    groups: Hashable | None = None,
    map_label: Callable[[DataArray], DataArray] = lambda x: x,
) -> Dataset:
    """
    Add splits/folds to a dataset based on stratified outcome.
    When groups is specified, it will use StratifiedGroupKFold, which attempts to create
    folds which preserve the percentage of samples for each class as much as possible
    given the constraint of non-overlapping groups between splits. If groups is None
    it will use StratifiedKFold. You can also map the label via map_label to encode
    label into something more friendly for stratification, say encode continuous value as bins.
    """
    folds = np.full((ds.dims["index"], n_splits), "", dtype="O")
    cv_klass = StratifiedGroupKFold if groups is not None else StratifiedKFold

    for i, (train, test) in enumerate(
        cv_klass(n_splits=n_splits, random_state=seed, shuffle=True).split(
            X=ds["outcome"].values,
            y=np.asarray(map_label(ds["outcome"])),
            groups=get_split_groups(ds, groups).values if groups is not None else None,
        )
    ):
        folds[train, i] = "train"
        folds[test, i] = "test"
    return ds.assign(fold=(("index", "folds"), folds)).assign_coords(
        folds=np.arange(n_splits)
    )


def add_unlabeled_data(
    ds: Dataset,
    *,
    df: DataFrame,
    index: Sequence[Hashable],
    fold_label: str,
) -> Dataset:
    """
    Adds out-of-sample examples. These examples won't be used for training, but will
    be used during the test phase. These examples are expected to have missing outcome
    which will make these ineligible for evaluation, but can be predicted/explained.
    """
    if "folds" not in ds.dims:
        raise ValueError("Splits/folds must be added before out of sample examples")
    assert (
        df[ds["outcomes"]].isna().all().all()
    ), "Outcomes should be missing/NA for out-of-sample examples"
    ds_oos = create_dataset(
        df,
        index=index,
        outcomes=ds["outcomes"].values.tolist(),
        descriptors=ds["descriptors"].values.tolist(),
        outcome_type=ds.attrs["outcome_type"],
    )
    return ds.pipe(lambda ds: xr.merge([ds, ds_oos])).pipe(
        lambda ds: ds.assign(fold=ds["fold"].fillna(fold_label))
    )


def explain(
    ds: Dataset,
    *,
    model: str,
    method: Literal["tree-shap", "linear-shap", "linear-hamard"] = "tree-shap",
    shap_mode: Literal["true-to-model", "true-to-data"] | None = None,
    tree_shap_check_additivity: bool = True,
    **kwargs: Any,
) -> Dataset:
    return (
        ds
        # Assign a variable with shape (index,) containing the fold index
        # for which each sample was held out
        .assign(test_fold=lambda ds: (ds.fold == "test").argmax(dim="folds"))
        # Run explanations for each group of held-out samples
        .groupby("test_fold")
        .map(
            lambda ds: ds.pipe(
                _explain,
                model=model,
                method=method,
                shap_mode=shap_mode,
                tree_shap_check_additivity=tree_shap_check_additivity,
                **kwargs,
            )
        )
        .pipe(lambda ex_ds: ds.merge(ex_ds))
    )


def extract_estimator(pipeline: Pipeline) -> BaseEstimator:
    """
    Extract an estimator from a pipeline, assuming it is the final step.
    """
    est = pipeline[-1]
    if hasattr(est, "best_estimator_"):
        est = est.best_estimator_
    return est


def explain_tree_prediction(
    est: BaseEstimator,
    features: DataFrame,
    shap_mode: Literal["true-to-model", "true-to-data"] | None,
    tree_shap_check_additivity: bool,
    labels: npt.ArrayLike | None = None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    if shap_mode is None or shap_mode == "true-to-model":
        data = shap.sample(features, nsamples=1000)
        feature_perturbation = "interventional"
    else:
        data = None
        feature_perturbation = "tree_path_dependent"
    explanation = shap.TreeExplainer(
        est, data=data, feature_perturbation=feature_perturbation, **kwargs
    )(features, y=labels, check_additivity=tree_shap_check_additivity)
    shap_values, shap_base_values = explanation.values, explanation.base_values

    # The shape of `shap_values` depends on the model output used for the explanation
    assert shap_values.ndim in (2, 3)
    if shap_values.ndim == 3:
        if shap_values.shape[2] != 2:
            raise ValueError("Only binary outcomes currently supported")
        shap_values = shap_values[..., 1]
    assert (
        shap_values.shape == features.shape
    ), f"Expected shape {features.shape}, got {shap_values.shape} instead"

    # The shape of `shap_base_values` also depends on the model output used for the explanation
    if shap_base_values.ndim == 2:
        if shap_base_values.shape[1] != 2:
            raise ValueError("Only binary outcomes currently supported")
        shap_base_values = shap_base_values[..., 1]
    assert shap_base_values.shape == (
        features.shape[0],
    ), f"Expected shape {(features.shape[0],)}, got {shap_base_values.shape} instead"

    return shap_values, shap_base_values


def explain_linear_prediction(
    est: BaseEstimator,
    features: DataFrame,
    shap_mode: Literal["true-to-model", "true-to-data"] | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    if shap_mode is None or shap_mode == "true-to-model":
        masker = Independent(features, max_samples=1000)
    else:
        masker = features
    explanation = shap.LinearExplainer(est, masker=masker, **kwargs)(features)
    shap_values, shap_base_values = explanation.values, explanation.base_values
    assert (
        shap_values.ndim == 2
    ), f"Expected 2D shap values, got {shap_values.shape} instead"
    return shap_values, shap_base_values


def explain_linear_hamard_prediction(
    features: DataFrame,
    coefficients: npt.NDArray[np.float_],
    intercept: float,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    if np.squeeze(coefficients).ndim != 1:
        raise ValueError(
            f"Could not coerce coefficients array of shape {coefficients.shape} to 1D"
        )
    coefficients = np.squeeze(coefficients)
    if coefficients.shape[0] != features.shape[1]:
        raise ValueError(
            f"Number of coefficients ({coefficients.shape[0]}) does not equal number of features ({features.shape[1]})"
        )
    shap_values = features.values * coefficients[np.newaxis, :]
    shap_base_values = np.full(shap_values.shape[0], intercept)
    assert (
        shap_values.ndim == 2
    ), f"Expected 2D shap values, got {shap_values.shape} instead"
    return shap_values, shap_base_values


def _explain(
    ds: Dataset,
    *,
    model: str,
    method: Literal["tree-shap", "linear-shap", "linear-hamard"],
    shap_mode: Literal["true-to-model", "true-to-data"] | None,
    tree_shap_check_additivity: bool,
    **kwargs: Any,
) -> Dataset:
    def get_shap_values(ds: Dataset) -> Dataset:
        # NOTE: labels can be NaN here, and will be included
        X, Y = _split(ds, squeeze_y=True)

        if Y.ndim > 1:
            raise NotImplementedError(
                f"Explanations not supported for multi-task outcomes (Y.shape={Y.shape})"
            )

        # Transform the predictors manually so that shap can be provided the
        # underlying model directly (i.e. the final pipeline step)
        pipeline = ds.model.item(0).estimator

        shap_features = X
        if len(pipeline.steps) > 1:
            transform = Pipeline(pipeline.steps[:-1])
            transformed_features = transform.transform(shap_features)
            if not isinstance(transformed_features, pd.DataFrame):
                raise ValueError(
                    f"Pipeline for model {model} did not produce a DataFrame after transform. "
                    f"This is necessary for explanations; model pipeline:\n{pipeline}"
                )
            if (
                invalid_features := transformed_features.columns.difference(
                    shap_features.columns
                )
            ).size > 0:
                raise ValueError(
                    f"Pipeline for model {model} produced {len(invalid_features)} invalid features: {invalid_features}"
                )
            shap_features = transformed_features

        # Resolve final estimator in pipeline
        est = extract_estimator(pipeline)

        # Branch based on method for generating explanations
        if method == "tree-shap":
            shap_values, shap_base_values = explain_tree_prediction(
                est, shap_features, shap_mode, tree_shap_check_additivity, Y, **kwargs
            )
        elif method == "linear-shap":
            shap_values, shap_base_values = explain_linear_prediction(
                est, shap_features, shap_mode, **kwargs
            )
        elif method == "linear-hamard":
            # Only supports single outcome ATM
            shap_values, shap_base_values = explain_linear_hamard_prediction(
                shap_features, est.coef_, est.intercept_
            )
        else:
            raise ValueError(f'Method "{method}" not valid')

        # Reindex SHAP features and values to align to original features, filling in
        # any features dropped by the current estimator with NaNs
        shap_values = (
            pd.DataFrame(
                shap_values, index=shap_features.index, columns=shap_features.columns
            )
            .reindex_like(X)
            .values
        )
        shap_features = shap_features.reindex_like(X).values
        assert shap_values.shape == shap_features.shape == X.shape

        if not np.all((a := np.asarray(shap_base_values)) == a[0]):
            raise NotImplementedError(
                f"Multiple SHAP intercepts/base_values not implemented (base_values={shap_base_values})"
            )

        return xr.Dataset(
            data_vars={
                "shap_feature": (("index", "features"), shap_features),
                "shap_value": (("index", "features"), shap_values),
                "shap_base_value": (("index",), shap_base_values),
            },
            coords=ds.feature.coords,
        )

    return get_shap_values(
        ds.sel(folds=ds.test_fold.item(0), models=model)
    ).expand_dims("models")


def shap_explanation(ds: Dataset, index_col: str | None = "index") -> Explanation:
    """
    Creates SHAP Explanation object based on the given model Dataset. If you
    need to create Explanation per fold or model, do all the necessary filters
    on the dataset object before passing it in this function.
    """
    if not all(i in ds for i in ["shap_base_value", "shap_feature", "shap_value"]):
        raise ValueError(
            "Dataset must contain shap_base_value, shap_feature and shap_value,"
            " which are produced by the `explain` function."
        )
    return Explanation(
        values=ds["shap_value"].values,
        base_values=ds["shap_base_value"].values,
        data=ds["shap_feature"].values,
        feature_names=ds["features"].values,
        # This can often cause bugs in certain shap functions when this is not
        # a single array of strings, as would be the case with a multi-index,
        # so setting it to None will avoid using sample labels at all.
        display_data=None if index_col is None else ds[index_col].values,
    )


def add_models(
    ds: Dataset, estimator_fn: Callable[..., list[Model]], **kwargs: Any
) -> Dataset:
    """
    Add models to fit to a dataset. Any kwargs if present are passed into the
    estimator_fn function.
    """
    if "folds" not in ds.dims:
        raise ValueError("Splits/folds must be added before models")
    if "feature_names" in inspect.signature(estimator_fn).parameters:
        kwargs = {**{"feature_names": ds.features.values.tolist()}, **kwargs}
    estimators = [estimator_fn(**kwargs) for _ in range(ds.dims["folds"])]
    return ds.assign(estimator=(("folds", "estimators"), estimators)).assign(
        estimators=[est.slug for est in estimators[0]]
    )


def score_regression(ds: Dataset) -> Dataset:
    """Compute continuous outcome metrics."""
    scores = get_score_names(get_regression_scores)

    def _score(
        y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        mask = ~np.isnan(y_true)
        y_true, y_pred = y_true[mask], y_pred[mask]
        values = get_regression_scores(y_true, y_pred)
        return np.array([values[m] for m in scores])

    return (
        # Use the rows as the input core dims to make sure that each
        # call to `score` is provided predictions for a single
        # model + fold + outcome prediction
        xr.apply_ufunc(
            _score,
            ds.outcome,
            ds.prediction,
            vectorize=True,
            input_core_dims=[["index"], ["index"]],
            output_core_dims=[["scores"]],
        )
        .rename("score")
        .to_dataset()
        .assign(scores=scores)
    )


def score_classification(ds: Dataset) -> Dataset:
    """Compute binary classifier metrics."""
    scores = get_score_names(get_classification_scores)

    def _score(
        y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        mask = ~np.isnan(y_true)
        y_true, y_pred = y_true[mask], y_pred[mask]
        values = get_classification_scores(y_true, y_pred)
        return np.array([values[m] for m in scores])

    return (
        # Use the rows as the input core dims to make sure that each
        # call to `score` is provided predictions for a single
        # model + fold + outcome prediction
        xr.apply_ufunc(
            _score,
            ds.outcome,
            ds.prediction.sel(classes="positive"),
            vectorize=True,
            input_core_dims=[["index"], ["index"]],
            output_core_dims=[["scores"]],
        )
        .rename("score")
        .to_dataset()
        .assign(scores=scores)
    )


def run_cv(
    ds: Dataset,
    predict_fn: Callable[[Dataset], Dataset],
    score_fn: Callable[[Dataset], Dataset],
    groups: Hashable | None = None,
) -> Dataset:
    """Run cross-validation for models and folds already defined in a dataset."""
    res = []

    for fold in tqdm.tqdm(sorted(ds.coords["folds"])):
        # Select training data, fit, and predict
        split = ds.fold.sel(folds=fold).values
        train, test, nans = split == "train", split == "test", pd.isnull(split)
        assert train.sum() + test.sum() + nans.sum() == ds.dims["index"]

        # Subset rows to those for training split and fit
        with warnings.catch_warnings():
            # Ignore convergence warnings from lbfgs
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            ds_train = (
                ds.sel(folds=fold)
                .sel(index=train)
                .pipe(fit, groups=groups)[["model", "outcome"]]
            )

        # Subset rows to those for test split and make predictions
        ds_test = (
            ds.sel(folds=fold)
            .sel(index=test)
            .assign(model=ds_train["model"])
            .pipe(predict_fn)[["prediction", "model", "outcome"]]
        )
        # Evaluate test set predictions
        ds_score = score_fn(ds_test)

        # Save test split results and scores
        res.append(
            # Note: scores do not contain row dim
            ds_test.merge(ds_score)
        )

    return ds.merge(xr.concat([r[["prediction"]] for r in res], dim="index")).merge(
        xr.concat([r[["model", "score"]] for r in res], dim="folds")
    )


def run_classification_cv(ds: Dataset, groups: Hashable | None = None) -> Dataset:
    """Run cross-validation for classification models and folds already defined in a dataset."""
    return run_cv(
        ds, predict_fn=predict_proba, score_fn=score_classification, groups=groups
    )


def run_regression_cv(ds: Dataset, groups: Hashable | None = None) -> Dataset:
    """Run cross-validation for regression models and folds already defined in a dataset."""
    return run_cv(ds, predict_fn=predict, score_fn=score_regression, groups=groups)


def create_dataset(
    df: DataFrame,
    index: Sequence[Hashable],
    outcomes: Sequence[Hashable],
    descriptors: Sequence[Hashable] | None = None,
    outcome_type: str | None = None,
) -> Dataset:
    """Create a dataset from a feature DataFrame using a partitioning of column names.

    Parameters
    ----------
    df
        Data frame to split into separate DataArray instances in one Dataset
    index
        List of column names to use as row index
    outcomes
        List of column names to use as separate outcomes (one outcome for a single-task regression)
    descriptors
        Optional list of column names with ancillary information; defaults to anything
        not in one of the above lists
    outcome_type
        Optional outcome type; defaults to the result of `sklearn.utils.multiclass.type_of_target`.
        See sklearn for possible results, additionally allows `regression:aft`.

    Returns
    -------
    ds : Dataset
        A dataset containing all features as one numeric DataArray (indexed by the provided
        index fields), an outcomes array, and a descriptors array.
    """
    if len(index) == 0:
        raise ValueError("Index fields list cannot be empty")
    if len(outcomes) == 0:
        raise ValueError("Outcomes field list cannot be empty")
    descriptors = descriptors or []

    # Set features as any numeric field not otherwise classified
    features = [
        c
        for c in df.select_dtypes(include=np.number).columns.tolist()
        if c not in (list(outcomes) + list(index) + list(descriptors))
    ]

    # Include all descriptor fields specified as well as those that are not numeric
    descriptors = list(
        set(
            list(descriptors)
            + [c for c in df if c not in list(features) + list(outcomes) + list(index)]
        )
    )
    if len(descriptors) == 0:
        # If there's no descriptors, we cook one up and delete it later,
        # makes for a cleaner code. Surprisingly some xarray function do not
        # work on "empty" inputs, including concat and transpose.
        del_descriptor_dim = True
        assert "__create_dataset_deleteme__" not in df
        df = df.assign(__create_dataset_deleteme__=np.NaN)
        descriptors = ["__create_dataset_deleteme__"]
    else:
        del_descriptor_dim = False

    # Ensure that no one column is present in multiple groups
    variables = pd.Series(
        list(index) + list(features) + list(outcomes) + list(descriptors)
    )
    if variables.duplicated().any():
        raise ValueError(
            f"Field partitioning invalid, found duplicated variables: {variables[variables.duplicated()].tolist()}"
        )

    ds = (
        df
        # Convert raw feature dataframe to xarray dataset with target index
        .to_xarray().set_index(index=index)
        # Group features, outcomes, and everything else into separate, individual data arrays
        .assign(
            # Group all numeric feature values into one 64-bit float array
            feature=lambda ds: xr.concat(
                [ds[c] for c in features], dim="features"
            ).astype("f8"),
            # Group all the regression outcomes into one array
            outcome=lambda ds: xr.concat([ds[c] for c in outcomes], dim="outcomes"),
            # Group all textual or otherwise descriptive fields into one array
            descriptor=lambda ds: xr.concat(
                [ds[c] for c in descriptors], dim="descriptors"
            ),
        )
        # Add additional indexes for new dimensions
        .assign(
            {"features": features, "outcomes": outcomes, "descriptors": descriptors}
        )
        # Subset to remove original feature fields
        .pipe(lambda ds: ds[["feature", "outcome", "descriptor"]])
        # Re-orient for consistency
        .transpose("index", "features", "outcomes", "descriptors")
    )

    if outcome_type is None:
        target_type = type_of_target(ds.outcome.squeeze().values)
        if target_type in ["binary", "multilabel-indicator"]:
            outcome_type = f"classification:{target_type}"
        elif target_type in ["continuous", "continuous-multioutput"]:
            outcome_type = f"regression:{target_type}"
        if outcome_type is None:
            raise ValueError(f"Outcome type {target_type} not supported")

    if del_descriptor_dim:
        ds = ds.drop_dims("descriptors")

    return ds.assign_attrs(outcome_type=outcome_type)


#########################
# Performance Functions #
#########################


def precision_at_k(y_true: Any, y_score: Any, k: int) -> float:
    """
    Compute precision at rank `k`

    This is defined as the fraction of examples that are positive
    amongst the top k predictions.
    """
    assert np.isin(y_true, [0, 1]).all()
    # Sort, reverse, limit to first k
    o = np.argsort(y_score)[::-1][: min(k, len(y_true))]
    # Find fraction true
    return float(y_true[o].mean())


def recall_at_k(y_true: Any, y_score: Any, k: int) -> float:
    """
    Compute recall at rank `k`

    This is defined as the ratio of positive examples in the top k
    predictions to the number of positive examples total.
    """
    assert np.isin(y_true, [0, 1]).all()
    # Sort, reverse, limit to first k
    o = np.argsort(y_score)[::-1][: min(k, len(y_true))]
    return float(y_true[o].sum() / y_true.sum())


def get_classification_scores(
    y_true: Any,
    y_score: Any,
    k: Sequence[int] = [10, 50, 100, 200, 500, 1000],
    threshold: float = 0.5,
) -> dict[str, int | float]:
    """
    Calculate performance of a classifier for a variety of metrics.

    Parameters:
    - y_true: array with binary outcomes.
    - y_score: predictions, assumed to be probabilities.
    - k: ranks to use for precision_at_k and recall_at_k.
    - threshold: threshold to use to convert y_score into binary predictions.

    General metrics:
    - n: number of samples
    - support: number of positive samples
    - balance: proportion of positive samples

    Metrics based on binary predictions at the given threshold:
    - precision
    - recall
    - f1
    - mcc: Matthews correlation coefficient

    Metrics based on continuous predictions (rank only):
    - roc_auc:
      Area under the receiver operator characteristic curve.
      Probability that a positive received a higher prediction than a negative.
    - average_precision

    Metrics based on continuous predictions (assesses calibration):
    - brier
    - balanced_mae:
      Balanced mean absolute error, aka macro-averaged mean absolute error.
      How far are predictions from truth, weighted to balance classes.
    - log_loss

    Metrics based on binary predictions at the given ranks:
    - precision@k
    - recall@k
    """
    y_true = y_true.astype(float)  # in case it's bool
    y_pred_binary = (y_score > threshold).astype(int)
    balancer = y_true / y_true.sum() + (1 - y_true) / (1 - y_true).sum()
    y_true_nunique = len(np.unique(y_true))
    y_pred_nunique = len(np.unique(y_pred_binary))

    if y_true_nunique == 1 or y_pred_nunique == 1:
        with warnings.catch_warnings():
            # Ignore RuntimeWarning: invalid value encountered in double_scalars
            # mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in double_scalars"
            )
            mcc = matthews_corrcoef(y_true, y_pred_binary)
    else:
        mcc = matthews_corrcoef(y_true, y_pred_binary)

    return {
        # general metrics
        "n_samples": len(y_true),
        "n_positives": y_true.sum(),
        "balance": y_true.mean(),
        # metrics for score threshold
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary),
        "mcc": mcc,
        # metrics for variable thresholds
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "balanced_mae": mean_absolute_error(y_true, y_score, sample_weight=balancer),
        "log_loss": log_loss(y_true, y_score),
        # metrics for ranking thresholds
        **{f"precision@{i}": precision_at_k(y_true, y_score, i) for i in k},
        **{f"recall@{i}": recall_at_k(y_true, y_score, i) for i in k},
    }


def get_regression_scores(
    y_true: Any,
    y_pred: Any,
) -> dict[str, int | float]:
    """
    Calculate performance of a regressor for a variety of metrics.

    Parameters:
    - y_true: array with binary outcomes.
    - y_pred: predictions

    """
    return {
        "n_samples": len(y_true),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def get_classification_score_names() -> list[str]:
    """Get names of all metrics returned by `get_classification_scores`"""
    return get_score_names(get_classification_scores)


def get_regression_score_names() -> list[str]:
    """Get names of all metrics returned by `get_regression_scores`"""
    return get_score_names(get_regression_scores)


def get_score_names(
    fn: Any,
    test_y_true: npt.NDArray[np.int_] = np.array([1, 0]),  # noqa: B008
    test_y_pred: npt.NDArray[np.int_] = np.array([1, 0]),  # noqa: B008
) -> list[str]:
    """Get names of all metrics returned by a scoring function for a particular outcome type"""
    return list(fn(test_y_true, test_y_pred))


def save(ds: xr.Dataset, path: Path) -> Path:
    """Save an xrml Xarray Dataset"""
    # Split evaluation/training data from trained model instances
    ds_data = (
        ds.drop_vars(["model", "estimator"], errors="ignore")
        # Drop any multi-indexes since they cannot be serialized and preserve the variable names
        # for those indexes so that they can be recovered on reload
        .pipe(
            lambda ds: ds.assign_attrs(
                index_names=json.dumps({"index": list(ds.indexes["index"].names)})
            )
        ).reset_index("index")
        # NOTE: otherwise nan is serialized as empty string, which causes problems down
        #       the line, the read method will revert this.
        .assign(fold=lambda ds: ds["fold"].fillna("__missing__"))
        # Convert this array of objects to string explicitly to avoid serialization errors
        # NOTE: Using vectorized string conversion to avoid strange error
        # `ValueError: setting an array element with a sequence` with .astype('U')
        .assign(
            descriptor=lambda ds: (
                ds.descriptor.dims,
                np.vectorize(str)(ds["descriptor"]),
            )
        )
    )
    # Serialize models as pickle and write everything else to Zarr store
    ds_data.to_zarr(
        fsspec.get_mapper(path.joinpath("data.zarr")),
        mode="w",
        consolidated=True,
    )
    if "model" in ds:
        ds_model = ds[["model"]]
        with fsspec.open(path.joinpath("models.pkl"), "wb") as f:
            dill.dump(ds_model, f)
    if "estimator" in ds:
        ds_est = ds[["estimator"]]
        with fsspec.open(path.joinpath("estimators.pkl"), "wb") as f:
            dill.dump(ds_est, f)
    return path


def load(path: Path, load_data: bool = True, load_models: bool = True) -> Dataset:
    """Reads artifact back into xr.Dataset"""
    assert load_data or load_models
    ds = xr.Dataset()
    if load_data:
        ds = ds.merge(
            xr.open_zarr(
                fsspec.get_mapper(path.joinpath("data.zarr")), consolidated=True
            )
            .pipe(lambda ds: ds.set_index(json.loads(ds.attrs["index_names"])))
            .assign(
                fold=lambda ds: ds["fold"].where(ds["fold"] != "__missing__", np.nan)
            )
        )
    models_url = path.joinpath("models.pkl")
    fs: AbstractFileSystem = get_filesystem_class(get_protocol(models_url))()
    if load_models:
        if not fs.exists(models_url):
            logger.warning(
                f"Model pickle files not present at {models_url!r}, skipping ..."
            )
        else:
            try:
                with fsspec.open(models_url) as f:
                    unpickled = dill.load(f)
            except BaseException as e:
                logger.warning(
                    "Couldn't load models, likely due to version mismatch, skipping ...",
                    e,
                )
            else:
                ds = ds.merge(unpickled)
        estimators_url = path.joinpath("estimators.pkl")
        if not fs.exists(estimators_url):
            logger.warning(
                f"Estimator pickle files not present at {estimators_url!r}, skipping ..."
            )
        else:
            try:
                with fsspec.open(estimators_url) as f:
                    unpickled = dill.load(f)
            except BaseException as e:
                logger.warning(
                    "Couldn't load estimators, likely due to version mismatch, skipping ...",
                    e,
                )
            else:
                ds = ds.merge(unpickled)
    return ds
