import numpy as np
from scipy.misc import derivative


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label

    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return -(a * t + (1 - a) * (1 - t)) * ((1 - (t * p + (1 - t) * (1 - p))) ** g) * (
                    t * np.log(p) + (1 - t) * np.log(1 - p))

    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)

    return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label
    p = 1 / (1 + np.exp(-y_pred))
    loss = -(a * y_true + (1 - a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g) * (
                y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    return 'focal_loss', np.mean(loss), False