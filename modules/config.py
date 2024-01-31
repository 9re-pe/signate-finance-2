class DirFile:
    root = "/Users/taichi/Library/CloudStorage/OneDrive-個人用/Documents/project/signate-finance-2/"

    data = root + "input/data/"
    train = data + "train.csv"
    test = data + "test.csv"
    submit = data + "sample_submission.csv"

    external = root + "input/external/"
    state_latlng = external + "state_latlng.csv"
    city_latlng = external + "city_latlng.csv"

    output = root + "output/"


class Params:
    seed = 42
    fold_num = 5
    lgb_learning_rate = 0.01
    lgb_max_depth = 6
    lgb_n_estimators = 1000
    lgb_colsample_bytree = 0.7
    lgb_stopping_rounds = 200
    lgb_verbose_eval = 0


class Cols:
    target = "MIS_Status"
    fold = "fold"
    sub_idx = "idx"
    sub_pred = "pred_class"
    weight = "loss_weight"
