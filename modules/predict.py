import numpy as np

from . import config as cfg


def predict_probability(models, test, drop_cols: list = None):
    """予測値の算出（lgb.train モデル用）"""

    if drop_cols is None:
        drop_cols = []

    test_feat_df = test.drop([cfg.Cols.fold, cfg.Cols.target] + drop_cols)
    pred_prob = np.array([model.predict(test_feat_df.to_numpy(), num_iteration=model.best_iteration) for model in models])
    pred_prob = np.mean(pred_prob, axis=0)

    return pred_prob


def predict_class(pred_prob, threshold: float = .5):
    """予測値のクラス化"""

    pred_class = np.where(pred_prob > threshold, 1, 0)

    return pred_class
