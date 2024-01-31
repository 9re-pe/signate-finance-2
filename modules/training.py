import numpy as np
import polars as pl
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from bokbokbok.loss_functions.classification import WeightedCrossEntropyLoss, WeightedFocalLoss
from bokbokbok.eval_metrics.classification import WeightedCrossEntropyMetric, WeightedFocalMetric
from bokbokbok.utils import clip_sigmoid

from . import config as cfg
from .utils import Timer
from . import preprosess
from . import predict
from . import metrics


def fit_lgbm(train, lgb_params: dict = None, zero_weight=0.5, drop_cols: list = None):
    """LightGBM + Sampling weight + CVによる学習を行う"""

    if lgb_params is None:
        lgb_params = {}
    if drop_cols is None:
        drop_cols = []

    # 扱いやすいようにpandasに変換
    train = train.to_pandas()

    # CVインデックス作成
    cv = []
    for fold in range(cfg.Params.fold_num):
        idx_train = np.array(train[train['fold'] != fold].index.to_list())
        idx_valid = np.array(train[train['fold'] == fold].index.to_list())
        cv.append((idx_train, idx_valid))

    # 特徴量と目的変数を抽出
    X = train.drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols, axis=1).values
    y = train[cfg.Cols.target].values

    models = []
    oof = np.zeros((len(train),), dtype=np.float32)
    for fold, (idx_train, idx_valid) in enumerate(cv):
        print(f"{'-' * 80}")
        print(f"START fold {fold + 1}")

        # split data
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        # sample weight
        train_weights = np.zeros(len(y_train))
        train_weights[y_train == 0] = zero_weight
        train_weights[y_train == 1] = 1.0 - zero_weight
        valid_weights = np.ones(len(x_valid))

        # fitting
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        with Timer(prefix=f"Time: "):
            lgb_model.fit(x_train, y_train,
                          sample_weight=train_weights,
                          eval_set=[(x_valid, y_valid)],
                          eval_metric=LgbCustomMetrics.lgb_macro_f1,
                          eval_sample_weight=[valid_weights],
                          callbacks=[
                              lgb.early_stopping(stopping_rounds=200, verbose=True),
                              lgb.log_evaluation(cfg.Params.lgb_verbose_eval)
                          ]
                          )

        # predict out-of-fold
        oof[idx_valid] = lgb_model.predict_proba(x_valid)[:, 1]
        models.append(lgb_model)

    print("=" * 80)
    print("FINISH!")

    return oof, models


def fit_lgbm_fl(train, drop_cols: list = None, lgb_params: dict = None, alpha: float = 0.25, gamma: float = 1.):
    """LightGBM + CVによる学習を行う(Focal Loss用)"""

    # 引数例外処理
    if drop_cols is None:
        drop_cols = []
    if lgb_params is None:
        lgb_params = {}

    models = []
    n_records = len(train)
    oof = np.zeros((n_records,), dtype=np.float32)

    # polarsはインデックスの概念がないのでカラムとして付与
    index_col = "idx"
    train_with_index = train.with_row_count(index_col)

    for fold in range(cfg.Params.fold_num):
        print(f"{'-' * 80}")
        print(f"START fold {fold + 1}")

        # split data
        x_train = train.filter(pl.col(cfg.Cols.fold) != fold).drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols)
        y_train = train.filter(pl.col(cfg.Cols.fold) != fold)[cfg.Cols.target].to_numpy()
        x_valid = train.filter(pl.col(cfg.Cols.fold) == fold).drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols)
        y_valid = train.filter(pl.col(cfg.Cols.fold) == fold)[cfg.Cols.target].to_numpy()
        idx_valid = train_with_index.filter(pl.col(cfg.Cols.fold) == fold)[index_col].to_numpy()

        # Convert data to lightgbm.Dataset
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_valid = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

        # fitting
        with Timer(prefix=f"Time: "):
            lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                fobj=WeightedFocalLoss(alpha=alpha, gamma=gamma),
                feval=WeightedFocalMetric(alpha=alpha, gamma=gamma),
                callbacks=[
                    lgb.early_stopping(stopping_rounds=cfg.Params.stopping_rounds),
                    lgb.log_evaluation(100)
                ]
            )

        # predict out-of-fold
        oof[idx_valid] = clip_sigmoid(lgb_model.predict(x_valid, num_iteration=lgb_model.best_iteration))
        models.append(lgb_model)

    print("=" * 80)
    print("FINISH!")

    return oof, models


def fit_lgbm_wcel(train, drop_cols: list = None, lgb_params: dict = None, alpha: float = 0.25):

    # 引数例外処理
    if drop_cols is None:
        drop_cols = []
    if lgb_params is None:
        lgb_params = {}

    models = []
    n_records = len(train)
    oof = np.zeros((n_records,), dtype=np.float32)

    # polarsはインデックスの概念がないのでカラムとして付与
    index_col = "idx"
    train_with_index = train.with_row_count(index_col)

    for fold in range(cfg.Params.fold_num):
        print(f"{'-' * 80}")
        print(f"START fold {fold + 1}")

        # split data
        x_train = train.filter(pl.col(cfg.Cols.fold) != fold).drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols)
        y_train = train.filter(pl.col(cfg.Cols.fold) != fold)[cfg.Cols.target].to_numpy()
        x_valid = train.filter(pl.col(cfg.Cols.fold) == fold).drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols)
        y_valid = train.filter(pl.col(cfg.Cols.fold) == fold)[cfg.Cols.target].to_numpy()
        idx_valid = train_with_index.filter(pl.col(cfg.Cols.fold) == fold)[index_col].to_numpy()

        # Convert data to lightgbm.Dataset
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_valid = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

        # fitting
        with Timer(prefix=f"Time: "):
            lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                fobj=WeightedCrossEntropyLoss(alpha=alpha),
                feval=WeightedCrossEntropyMetric(alpha=alpha),
                callbacks=[
                    lgb.early_stopping(stopping_rounds=cfg.Params.stopping_rounds),
                    lgb.log_evaluation(100)
                ]
            )

        # predict out-of-fold
        oof[idx_valid] = clip_sigmoid(lgb_model.predict(x_valid, num_iteration=lgb_model.best_iteration))
        models.append(lgb_model)

    print("=" * 80)
    print("FINISH!")

    return oof, models


def show_feature_importance(models, train, drop_cols=None):
    """lightGBM の model 配列の feature importance を plot する（lgb.train モデル用）"""

    if drop_cols is None:
        drop_cols = []

    if cfg.Cols.weight in train.columns:
        drop_cols.append(cfg.Cols.weight)

    # pandasに変換
    feat_train_df = train.to_pandas().drop([cfg.Cols.target, cfg.Cols.fold] + drop_cols, axis=1)
    feature_importance_df = pd.DataFrame()

    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importance(importance_type='gain')  # gain を使用
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby("column") \
                .sum()[["feature_importance"]] \
                .sort_values("feature_importance", ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * .25)))
    sns.boxenplot(
        data=feature_importance_df,
        x="feature_importance",
        y="column",
        order=order,
        ax=ax,
        orient="h"
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()

    return fig, ax


def fit_stacking(train, oof_li, lr_params: dict = None):
    """スタッキング用のモデルを学習する"""

    # 引数例外処理
    if lr_params is None:
        lr_params = {}

    models = []
    n_records = len(train)
    oof = np.zeros((n_records,), dtype=np.float32)

    # スタッキング用の特徴量を作成（各モデルの予測値を結合）
    stacked_features = np.column_stack(oof_li)

    # polarsはインデックスの概念がないのでカラムとして付与
    index_col = "idx"
    train_with_index = train.with_row_count(index_col)

    for fold in range(cfg.Params.fold_num):
        print(f"{'-' * 80}")
        print(f"START fold {fold + 1}")

        # split data for stacking
        idx_train = train_with_index.filter(pl.col(cfg.Cols.fold) != fold)[index_col].to_numpy()
        idx_valid = train_with_index.filter(pl.col(cfg.Cols.fold) == fold)[index_col].to_numpy()
        x_train = stacked_features[idx_train]
        y_train = train.filter(pl.col(cfg.Cols.fold) != fold)[cfg.Cols.target].to_numpy()
        x_valid = stacked_features[idx_valid]
        y_valid = train.filter(pl.col(cfg.Cols.fold) == fold)[cfg.Cols.target].to_numpy()

        # fitting logistic regression for stacking
        with Timer(prefix=f"Time: "):
            lr_model = LogisticRegression(**lr_params)
            lr_model.fit(x_train, y_train)

        # predict out-of-fold
        oof[idx_valid] = lr_model.predict_proba(x_valid)[:, 1]
        models.append(lr_model)

    print("=" * 80)
    print("FINISH!")

    return oof, models


class LgbCustomMetrics:
    @staticmethod
    def lgb_macro_f1(y_true, y_pred):
        y_pred = np.round(y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        return 'macroF1', f1, True

    @staticmethod
    def lgb_weight_macro_f1(y_true, y_pred, ratio):
        y_pred = np.round(y_pred)
        f1 = np.mean(f1_score(y_true, y_pred, average=None) * np.array(ratio))

        return 'macroF1', f1, True

    @staticmethod
    def lgb_macro_recall(y_true, y_pred):
        y_pred_labels = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
        ratio_list = [[0.2, 0.1, 0.7], [0.3, 0.1, 0.6], [0.4, 0.1, 0.5], [0.25, 0.15, 0.6], [1, 1, 1]]
        recall = sum(recall_score(y_true, y_pred_labels, average=None) * np.array(ratio_list[s]))

        return 'macro_recall', recall, True

    @staticmethod
    def lgb_macro_precision(y_true, y_pred):
        y_pred_labels = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
        ratio_list = [[0.2, 0.1, 0.7], [0.3, 0.1, 0.6], [0.4, 0.1, 0.5], [0.25, 0.15, 0.6], [1, 1, 1]]
        precision = sum(precision_score(y_true, y_pred_labels, average=None, zero_division=0) * np.array(ratio_list[s]))

        return 'macro_precision', precision, True
