import numpy as np
import polars as pl

from . import config as cfg
from . import training
from . import predict
from . import metrics


def tuning_loss_weight(train, lgb_params: dict = None, zero_weights: list = None, drop_cols: list = None):
    """loss_weightのチューニング"""

    threshold = 0.5
    if lgb_params is None:
        lgb_params = {}
    if zero_weights is None:
        zero_weights = np.arange(0.74, 0.88, 0.01)
    if drop_cols is None:
        drop_cols = []

    max_cv_score = -1.0
    best_zero_weight = 0.0
    # TODO: tqdmを使う + fit_lgbm()の出力を出さないようにする
    for zero_weight in zero_weights:
        oof, models = training.fit_lgbm(train, lgb_params, zero_weight, drop_cols)
        oof_truth = train[cfg.Cols.target].to_numpy()
        oof_hat = predict.predict_class(oof, threshold=threshold)
        cv_score = metrics.macro_f1_score(oof_truth, oof_hat)

        if cv_score > max_cv_score:
            max_cv_score = cv_score
            best_zero_weight = zero_weight

    return max_cv_score, best_zero_weight


def search_effective_feature(
        train,
        ignore_features: list = None,
        drop_cols: list = None,
        lgb_params: dict = None,
        zero_weights: list = None):
    """特徴量の効果を確認するために，特徴量を1つずつ削除してCVを計算する"""

    if drop_cols is None:
        drop_cols = []

    features = train.drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols).columns
    if ignore_features is not None:
        ignore_features = set(ignore_features)
        features = [feature for feature in features if feature not in ignore_features]

    results_df = pl.DataFrame([
        pl.Series("feature", [], dtype=pl.Utf8),
        pl.Series("cv_score", [], dtype=pl.Float64),
        pl.Series("zero_weight", [], dtype=pl.Float64),
        pl.Series("diff", [], dtype=pl.Float64),
    ])

    base_cv_score, base_zero_weight = tuning_loss_weight(
        train, lgb_params=lgb_params, zero_weights=zero_weights, drop_cols=drop_cols
    )
    results_df = results_df.vstack(pl.DataFrame({
        "feature": "base",
        "cv_score": [base_cv_score],
        "zero_weight": [base_zero_weight],
        "diff": [0.0],
    }))

    for feature in features:
        check_train = train.drop([feature])
        cv_score, zero_weight = tuning_loss_weight(
            check_train, lgb_params=lgb_params, zero_weights=zero_weights, drop_cols=drop_cols
        )
        diff = base_cv_score - cv_score
        results_df = results_df.vstack(pl.DataFrame({
            "feature": [feature],
            "cv_score": [cv_score],
            "zero_weight": [zero_weight],
            "diff": [diff],
        }))

    return results_df
