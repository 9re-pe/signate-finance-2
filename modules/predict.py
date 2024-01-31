import numpy as np
from bokbokbok.utils import clip_sigmoid

from . import config as cfg


def predict_probability(models, test, drop_cols: list = None):
    """予測値の算出"""

    if drop_cols is None:
        drop_cols = []

    x_test = test.drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols).to_numpy()
    pred_prob = np.array([model.predict_proba(x_test)[:, 1] for model in models])
    pred_prob = np.mean(pred_prob, axis=0)

    return pred_prob


def predict_probability_sigmoid(models, test, drop_cols: list = None):
    """シグモイド関数を経由して予測値の算出"""

    if drop_cols is None:
        drop_cols = []

    x_test = test.drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols).to_numpy()
    pred_prob = np.array(
        [clip_sigmoid(model.predict(x_test, num_iteration=model.best_iteration)) for model in models]
    )
    pred_prob = np.mean(pred_prob, axis=0)

    return pred_prob


def predict_probability_stacking(models, meta_test):
    """スタッキングにおける予測値の算出"""

    X_test = np.column_stack(meta_test)
    pred_prob = np.array([model.predict_proba(X_test)[:, 1] for model in models])
    pred_prob = np.mean(pred_prob, axis=0)

    return pred_prob


def predict_class(pred_prob, threshold: float = .5):
    """予測値のクラス化"""

    pred_class = np.where(pred_prob > threshold, 1, 0)

    return pred_class



