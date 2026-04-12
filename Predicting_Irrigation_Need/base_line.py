from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

try:
    from catboost import CatBoostClassifier, Pool  # type: ignore

    _HAS_CATBOOST = True
except ModuleNotFoundError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore
    Pool = None  # type: ignore
    _HAS_CATBOOST = False


TARGET_COL_DEFAULT = "Irrigation_Need"
ID_COL = "id"


def _resolve_data_dir(data_dir: str) -> Path:
    """
    Make data_dir robust to running from repo root or script dir.
    Tries:
      1) given path
      2) relative to this script's directory
    """
    p = Path(data_dir)
    if p.exists():
        return p
    script_rel = Path(__file__).resolve().parent / data_dir
    if script_rel.exists():
        return script_rel
    raise FileNotFoundError(
        f"找不到 data_dir={data_dir!r}。请确认目录存在，或用 --data_dir 指定正确路径。"
    )


def _infer_target_col(sample_sub: pd.DataFrame) -> str:
    if TARGET_COL_DEFAULT in sample_sub.columns:
        return TARGET_COL_DEFAULT
    non_id_cols = [c for c in sample_sub.columns if c != ID_COL]
    if len(non_id_cols) == 1:
        return non_id_cols[0]
    raise ValueError(
        f"无法从 sample_submission 推断目标列名。columns={list(sample_sub.columns)}"
    )


def _split_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in (ID_COL,):
        if c in num_cols:
            num_cols.remove(c)

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    for c in (target_col,):
        if c in cat_cols:
            cat_cols.remove(c)
    if ID_COL in cat_cols:
        cat_cols.remove(ID_COL)

    return num_cols, cat_cols


def _prepare_xy(
    train: pd.DataFrame, test: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    assert ID_COL in train.columns, f"train 缺少列 {ID_COL!r}"
    assert ID_COL in test.columns, f"test 缺少列 {ID_COL!r}"
    assert target_col in train.columns, f"train 缺少目标列 {target_col!r}"

    num_cols, cat_cols = _split_columns(train, target_col=target_col)
    feature_cols = num_cols + cat_cols

    X_train = train[feature_cols].copy()
    y_train = train[target_col].copy()
    X_test = test[feature_cols].copy()

    # Ensure CatBoost categorical features are strings and missing are consistent.
    if cat_cols:
        for col in cat_cols:
            X_train[col] = X_train[col].astype("string").fillna("__NA__")
            X_test[col] = X_test[col].astype("string").fillna("__NA__")

    return X_train, y_train, X_test, cat_cols


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: List[str],
    n_splits: int,
    seed: int,
    verbose: bool,
    backend: str,
) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    backend = backend.lower().strip()
    if backend not in {"catboost", "sklearn"}:
        raise ValueError("backend 只能是 'catboost' 或 'sklearn'")

    fold_scores: List[float] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if backend == "catboost":
            if not _HAS_CATBOOST:
                raise ModuleNotFoundError(
                    "当前环境未安装 catboost。请先执行：pip install catboost"
                )
            cat_features = (
                [X.columns.get_loc(c) for c in cat_cols] if cat_cols else None
            )
            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            valid_pool = Pool(X_va, y_va, cat_features=cat_features)

            model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="TotalF1",  # CatBoost 内部早停信号
                random_seed=seed,
                iterations=2000,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                task_type="GPU",
                devices="0",
                od_type="Iter",
                od_wait=200,
                verbose=False,
            )

            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            y_pred = model.predict(X_va).reshape(-1)
        else:
            # lightweight fallback baseline without catboost
            from sklearn.ensemble import HistGradientBoostingClassifier

            X_tr_enc = X_tr.copy()
            X_va_enc = X_va.copy()

            if cat_cols:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )
                X_tr_enc[cat_cols] = enc.fit_transform(X_tr_enc[cat_cols])
                X_va_enc[cat_cols] = enc.transform(X_va_enc[cat_cols])

            model = HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_depth=8,
                max_iter=300,
                random_state=seed,
            )
            model.fit(X_tr_enc, y_tr)
            y_pred = model.predict(X_va_enc)

        score = balanced_accuracy_score(y_va, y_pred)
        fold_scores.append(score)
        print(f"[CV] fold={fold}/{n_splits} balanced_accuracy={score:.6f}")

        if verbose:
            print(classification_report(y_va, y_pred, digits=4))

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    print(f"[CV] mean={mean_score:.6f} std={std_score:.6f}")
    return mean_score


def train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    seed: int,
    backend: str,
) -> np.ndarray:
    backend = backend.lower().strip()
    if backend not in {"catboost", "sklearn"}:
        raise ValueError("backend 只能是 'catboost' 或 'sklearn'")

    if backend == "catboost":
        if not _HAS_CATBOOST:
            raise ModuleNotFoundError(
                "当前环境未安装 catboost。请先执行：pip install catboost"
            )
        cat_features = (
            [X_train.columns.get_loc(c) for c in cat_cols] if cat_cols else None
        )
        train_pool = Pool(X_train, y_train, cat_features=cat_features)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=seed,
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            task_type="GPU",
            devices="0",
            verbose=200,
        )
        model.fit(train_pool)
        return model.predict(X_test).reshape(-1)

    from sklearn.ensemble import HistGradientBoostingClassifier

    X_tr_enc = X_train.copy()
    X_te_enc = X_test.copy()
    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        X_tr_enc[cat_cols] = enc.fit_transform(X_tr_enc[cat_cols])
        X_te_enc[cat_cols] = enc.transform(X_te_enc[cat_cols])

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        random_state=seed,
    )
    model.fit(X_tr_enc, y_train)
    return model.predict(X_te_enc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predicting Irrigation Need baseline (CatBoost preferred, sklearn fallback)"
    )
    parser.add_argument("--data_dir", type=str, default="Predicting_Irrigation_Need/data", help="数据目录，包含 train.csv/test.csv/sample_submission.csv")
    parser.add_argument("--n_splits", type=int, default=5, help="StratifiedKFold 折数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_path", type=str, default="Predicting_Irrigation_Need/baseline_catboost.csv", help="提交文件保存路径")
    parser.add_argument("--verbose_report", action="store_true", help="打印每折 classification_report（较长）")
    parser.add_argument(
        "--backend",
        type=str,
        default="catboost",
        choices=["catboost", "sklearn"],
        help="训练后端：catboost（推荐）或 sklearn（不依赖 catboost 的兜底 baseline）",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    data_dir = _resolve_data_dir(args.data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    for p in (train_path, test_path, sample_path):
        if not p.exists():
            raise FileNotFoundError(f"缺少文件：{p}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)
    target_col = _infer_target_col(sample_sub)

    X_train, y_train, X_test, cat_cols = _prepare_xy(train, test, target_col=target_col)
    print(f"[Info] train_shape={train.shape} test_shape={test.shape}")
    print(f"[Info] target_col={target_col} num_features={X_train.shape[1]-len(cat_cols)} cat_features={len(cat_cols)}")
    if args.backend == "catboost" and not _HAS_CATBOOST:
        print("[Warn] 当前环境未安装 catboost，将无法运行 catboost 后端。")
        print("       解决：pip install catboost")
        print("       或使用：--backend sklearn 先跑通 baseline。")

    run_cv(
        X=X_train,
        y=y_train,
        cat_cols=cat_cols,
        n_splits=args.n_splits,
        seed=args.seed,
        verbose=args.verbose_report,
        backend=args.backend,
    )

    # Train full and create submission
    preds = train_and_predict(
        X_train,
        y_train,
        X_test,
        cat_cols=cat_cols,
        seed=args.seed,
        backend=args.backend,
    )

    if ID_COL not in sample_sub.columns:
        raise ValueError(f"sample_submission 缺少列 {ID_COL!r}，columns={list(sample_sub.columns)}")
    if target_col not in sample_sub.columns:
        raise ValueError(f"sample_submission 缺少目标列 {target_col!r}，columns={list(sample_sub.columns)}")

    sub = sample_sub.copy()
    sub[target_col] = preds
    save_path = Path(args.save_path)
    sub.to_csv(save_path, index=False)
    print(f"[OK] Saved submission to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
