class DirFile:
    root = "/Users/taichi/Library/CloudStorage/OneDrive-個人用/Documents/project/signate-finance-2/"

    data = root + "input/data/"
    train = data + "train.csv"
    test = data + "test.csv"
    submit = data + "sample_submission.csv"

    external = root + "input/external/"
    state_latlon = external + "state_latlon.csv"
    city_latlon = external + "city_latlon.csv"


    output = root + "output/"


class Params:
    seed = 42
    fold_num = 5
    learning_rate = 0.2


class Cols:
    target = "MIS_Status"
    fold = "fold"
    sub_idx = "idx"
    sub_pred = "pred_class"
