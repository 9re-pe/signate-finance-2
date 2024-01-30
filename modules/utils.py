import random
import os
import numpy as np
import polars as pl
from time import time

from . import config as cfg


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_data(debug_mode=False):
    """csvデータ読み込み"""

    if debug_mode:
        n_chunk = 1000
        train = pl.read_csv(cfg.DirFile.train).lazy().limit(n_chunk).collect()
        test = pl.read_csv(cfg.DirFile.test).lazy().limit(n_chunk).collect()
        submit = pl.read_csv(cfg.DirFile.submit, has_header=False).lazy().limit(n_chunk).collect()
        submit.columns = [cfg.Cols.sub_idx, cfg.Cols.sub_pred]
    else:
        train = pl.read_csv(cfg.DirFile.train)
        test = pl.read_csv(cfg.DirFile.test)
        submit = pl.read_csv(cfg.DirFile.submit, has_header=False)
        submit.columns = [cfg.Cols.sub_idx, cfg.Cols.sub_pred]

    return train, test, submit


def make_submission(submit, filename, cv_score):
    """提出ファイル作成"""
    submit.write_csv(cfg.DirFile.output + f"submit_{filename}_{cv_score}.csv", has_header=False)


def ignore_user_warning():
    """LightGBM等のUserWarningを無視する"""

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)


# ref: Kaggleコード遺産 https://qiita.com/kaggle_grandmaster-arai-san/items/d59b2fb7142ec7e270a5
class Timer:
    def __init__(self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)
