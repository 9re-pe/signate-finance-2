from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import polars as pl

from . import config as cfg


def assign_stratified_k_fold(train_df, objective_col, ):
    """StratifiedKFoldによるfoldの割り当て"""

    skf = StratifiedKFold(
        n_splits=cfg.Params.fold_num,
        shuffle=True,
        random_state=cfg.Params.seed
    )

    fold_assignments = np.full(train_df.height, -1, dtype=int)
    for i, (_, valid_index) in enumerate(skf.split(train_df, train_df[objective_col])):
        fold_assignments[valid_index] = i
    train_df = train_df.with_columns(pl.Series("fold", fold_assignments))

    return train_df


def convert_city_to_latlon(df):
    """Cityカラムから緯度経度に変換"""

    latlon_df = pl.read_csv(cfg.DirFile.latlon)
    df = df.join(latlon_df, on="City", how="left")

    return df


def convert_date_to_year(df, cols):
    """日付データからYearだけ抽出して数値とする

    変換前 : 30-Sep-94
    変換後 : 1094
    """

    for col_name in cols:
        df = df.with_columns(
            pl.col(col_name).str.strptime(pl.Date, "%d-%b-%y").dt.year().alias(f"{col_name}_year")
        )

    return df


def convert_money_data(df, cols):
    """金額データの$を削除し，数値に変換

    変換前 : $80,000.00
    変換後 : 80000.00
    """

    for col_name in cols:
        df = df.with_columns(
            pl.col(col_name).str.replace_all(",", "").str.strip().str.slice(1, ).str.extract("^(.+)\.").cast(pl.Int64)
        )

    return df


def unify_same_sector(df):
    """同じSector(説明上)に異なるカテゴリ変数が振られているのでまとめる"""

    df = df.with_columns(
        pl.when(pl.col("Sector") == 32).then(31)
          .when(pl.col("Sector") == 33).then(31)
          .when(pl.col("Sector") == 45).then(44)
          .when(pl.col("Sector") == 49).then(48)
          .otherwise(pl.col("Sector"))
        .alias("Sector")
    )

    return df


def add_eq(df, eqs):
    """2つのカラムが一致するかを新しいカラムとして追加"""

    for eq in eqs:
        df = df.with_columns(
            ((pl.col(eq[0]) == pl.col(eq[1])).cast(pl.Int64)).alias(f"{eq[0]}_{eq[1]}_diff")
        )

    return df


def add_diff(df, diffs):
    """2つのカラムの差分を新しいカラムとして追加"""

    for diff in diffs:
        df = df.with_columns(
            (pl.col(diff[0]) - pl.col(diff[1])).alias(f"{diff[0]}_{diff[1]}_diff")
        )

    return df


def add_div(df, divs):
    """2つのカラムの割合を新しいカラムとして追加"""

    for div in divs:
        df = df.with_columns(
            (pl.col(div[0]) / pl.col(div[1])).alias(f"{div[0]}_{div[1]}_div")
        )

    return df


def add_diff_disbursement_with_approval(df):
    """DisbursementDateとApprovalDateの日数の差を追加"""

    df = df.with_columns([
        pl.col("DisbursementDate").str.strptime(pl.Date, "%d-%b-%y").alias("DisbursementDate_parsed"),
        pl.col("ApprovalDate").str.strptime(pl.Date, "%d-%b-%y").alias("ApprovalDate_parsed")
    ])

    df = df.with_columns(
        (pl.col("DisbursementDate_parsed") - pl.col("ApprovalDate_parsed")).dt.days().alias("Disbursement_Approval_diff")
    )

    df = df.drop(["DisbursementDate_parsed", "ApprovalDate_parsed"])

    return df


# TODO: リークの可能性があるのでKaggle本に合わせる
def target_encoding(train_df, test_df, cols):
    """ターゲットエンコーディング"""

    for col_name in cols:
        te_df = train_df.select(pl.col([col_name, cfg.Cols.fold, cfg.Cols.target]))
        train_target_encoding_df = pl.DataFrame()
        for fold in range(cfg.Params.fold_num):
            te_df_fold = te_df.filter(pl.col("fold") != fold)
            te_df_fold = te_df_fold.groupby(col_name).agg(
                pl.col(cfg.Cols.target).mean().alias(f"{col_name}_{cfg.Cols.target}_mean")
            )
            te_df_fold = te_df_fold.with_columns(pl.lit(fold).alias("fold").cast(pl.Int64))
            train_target_encoding_df = pl.concat([train_target_encoding_df, te_df_fold])
        test_target_encoding_df = te_df.groupby(col_name).agg(
            pl.col(cfg.Cols.target).mean().alias(f"{col_name}_{cfg.Cols.target}_mean"),
        )

        train_df = train_df.join(train_target_encoding_df, on=[col_name, "fold"], how="left")
        test_df = test_df.join(test_target_encoding_df, on=col_name, how="left")

    return train_df, test_df


def label_encoding(train_df, test_df, cols):
    """ラベルエンコーディング"""

    for col_name in cols:
        le = LabelEncoder()

        train_categorical_col = train_df.select(pl.col(col_name)).to_numpy().flatten()
        train_label_col = le.fit_transform(train_categorical_col)
        train_df = train_df.with_columns([pl.Series(train_label_col).alias(col_name)])

        test_categorical_col = test_df.select(pl.col(col_name)).to_numpy().flatten()
        test_label_col = le.transform(test_categorical_col)
        test_df = test_df.with_columns([pl.Series(test_label_col).alias(col_name)])

    return train_df, test_df
