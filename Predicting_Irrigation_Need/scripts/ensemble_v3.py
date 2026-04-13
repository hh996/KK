from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    print("请安装 catboost: pip install catboost")

from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier

TARGET_COL = "Irrigation_Need"
ID_COL = "id"
LABEL_MAP = {"High": 0, "Low": 1, "Medium": 2}
LABEL_LIST = ["High", "Low", "Medium"]
MODEL_DISPLAY = {"cat": "CatBoost", "lgb": "LightGBM", "xgb": "XGBoost"}


def _resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    if p.exists():
        return p
    for base in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
        candidate = base / data_dir
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 data_dir={data_dir!r}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["High_Score"] = (
        2 * (df["Soil_Moisture"] < 25)
        + 2 * (df["Rainfall_mm"] < 300)
        + 1 * (df["Temperature_C"] > 30)
        + 1 * (df["Wind_Speed_kmh"] > 10)
    )
    df["Low_Score"] = (
        2 * df["Crop_Growth_Stage"].isin(["Harvest", "Sowing"])
        + 1 * (df["Mulching_Used"] == "Yes")
    )
    df["Rule_Score"] = df["High_Score"] - df["Low_Score"]
    df["No_Mulch_Dry"] = ((df["Mulching_Used"] == "No") & (df["Soil_Moisture"] < 28)).astype(int)
    df["Growth_Season"] = df["Crop_Growth_Stage"] + "_" + df["Season"]
    return df


def prepare_data(data_dir: str = "data"):
    data_dir = _resolve_data_dir(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

    train = add_features(train)
    test = add_features(test)

    num_cols = [c for c in train.select_dtypes(include="number").columns if c != ID_COL]
    cat_cols = [c for c in train.select_dtypes(exclude="number").columns if c not in (TARGET_COL, ID_COL)]

    X_train = train[num_cols + cat_cols].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[num_cols + cat_cols].copy()

    for col in cat_cols:
        X_train[col] = X_train[col].astype("string").fillna("__NA__")
        X_test[col] = X_test[col].astype("string").fillna("__NA__")

    return X_train, y_train, X_test, cat_cols, sample_sub


def _label_encode_cats(dfs: list[pd.DataFrame], cat_cols: list[str]):
    dfs = [df.copy() for df in dfs]
    for c in cat_cols:
        if c not in dfs[0].columns:
            continue
        le = LabelEncoder()
        le.fit(pd.concat([df[c] for df in dfs], ignore_index=True).astype(str))
        for df in dfs:
            df[c] = le.transform(df[c].astype(str))
    return dfs


def train_and_get_proba(model, X_tr, y_tr, X_va, y_va, X_test, model_name, cat_feature_indices, cat_cols):
    if model_name == "cat":
        train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
        valid_pool = Pool(X_va, y_va, cat_features=cat_feature_indices)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=200, verbose=False)
        return model.predict_proba(X_va), model.predict_proba(X_test)

    X_tr_e, X_va_e, X_test_e = _label_encode_cats([X_tr, X_va, X_test], cat_cols)

    if model_name == "lgb":
        model.fit(X_tr_e, y_tr)
        return model.predict_proba(X_va_e), model.predict_proba(X_test_e)

    if model_name == "xgb":
        le = LabelEncoder()
        y_tr_num = le.fit_transform(y_tr)
        model.fit(X_tr_e, y_tr_num)
        return model.predict_proba(X_va_e), model.predict_proba(X_test_e)

    raise ValueError(f"未知 model_name: {model_name!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str,
                        default="Predicting_Irrigation_Need/outputs/submissions/ensemble_v3.csv")
    args = parser.parse_args()

    seed = args.seed

    # ============ 统一参数配置（字典形式，CatBoost / LightGBM / XGBoost） ============
    cat_params = {
        "iterations": 6000,
        "learning_rate": 0.022,
        "depth": 7,
        "l2_leaf_reg": 5.0,
        "random_strength": 2.0,
        "bagging_temperature": 1.2,
        "auto_class_weights": "Balanced",
        "border_count": 254,
        "one_hot_max_size": 10,
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "random_seed": seed,
    }

    # 注意：Windows 下使用 GPU 需要安装预编译的 GPU 版本 wheel (通常是 OpenCL)
    lgb_params = {
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 9,
        "num_leaves": 256,
        "class_weight": "balanced",
        "random_state": seed,
        "verbose": -1,
        "device": "gpu",         # 使用 GPU (OpenCL)
        "gpu_use_dp": True,      # 使用双精度，防止 Windows GPU 出现精度溢出/NaN
    }

    # XGBoost 2.x 用 device='cuda'，旧版回退 gpu_hist
    xgb_major = int(xgb.__version__.split(".")[0])
    xgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 8,
        "eval_metric": "mlogloss",
        "random_state": seed,
        "verbosity": 0,
        "tree_method": "hist",
        **({"device": "cuda"} if xgb_major >= 2 else {"tree_method": "gpu_hist"}),
    }

    # 打印参数
    print("=" * 70)
    print("  模型参数配置")
    print("=" * 70)
    for label, params in [
        ("CatBoost  [GPU]", cat_params),
        ("LightGBM  [GPU]", lgb_params),
        (f"XGBoost   [GPU, v{xgb.__version__}]", xgb_params),
    ]:
        print(f"\n  [{label}]")
        for k, v in params.items():
            print(f"    {k}: {v}")

    base_models = {
        # "cat": CatBoostClassifier(**cat_params),
        "lgb": LGBMClassifier(**lgb_params),
        # "xgb": XGBClassifier(**xgb_params),
    }

    X_train, y_train, X_test, cat_cols, sample_sub = prepare_data(args.data_dir)
    print(f"\n{'=' * 70}")
    print(f"  数据: X_train {X_train.shape} | cat_cols {len(cat_cols)} | 设备: CatBoost=GPU  LightGBM=GPU  XGBoost=GPU")
    print(f"{'=' * 70}")

    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

    n_classes = 3
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)

    oof = {k: np.zeros((len(y_train), n_classes)) for k in base_models}
    test_pred = {k: np.zeros((len(X_test), n_classes)) for k in base_models}
    y_num = y_train.map(LABEL_MAP)

    model_names = list(base_models.keys())
    total_tasks = args.n_splits * len(model_names)
    model_elapsed = {k: 0.0 for k in base_models}
    task_count = 0
    t_global_start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_num), 1):
        print(f"\n{'─' * 70}")
        print(f"  Fold {fold}/{args.n_splits}")
        print(f"{'─' * 70}")

        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        for name in model_names:
            task_count += 1
            model = clone(base_models[name])

            t0 = time.time()
            proba_va, proba_test = train_and_get_proba(
                model, X_tr, y_tr, X_va, y_va, X_test,
                name, cat_feature_indices, cat_cols,
            )
            elapsed = time.time() - t0
            model_elapsed[name] += elapsed

            oof[name][va_idx] = proba_va
            test_pred[name] += proba_test / args.n_splits

            # 当前折该模型效果
            va_pred = np.argmax(proba_va, axis=1)
            va_score = balanced_accuracy_score(y_va.map(LABEL_MAP), va_pred)

            # 该模型平均每折耗时 × 剩余折数
            avg_per_fold = model_elapsed[name] / fold
            est_model_remain = avg_per_fold * (args.n_splits - fold)

            # 全局：已完成任务的平均耗时 × 剩余任务数
            total_elapsed_so_far = sum(model_elapsed.values())
            est_global_remain = (total_elapsed_so_far / task_count) * (total_tasks - task_count)

            print(
                f"  {MODEL_DISPLAY[name]:>8} | "
                f"Fold {fold:>2}/{args.n_splits} | "
                f"BalAcc: {va_score:.6f} | "
                f"本轮: {elapsed:>6.1f}s | "
                f"累计: {model_elapsed[name]:>7.1f}s | "
                f"模型预计剩余: {est_model_remain:>6.1f}s | "
                f"全局预计剩余: {est_global_remain:>7.1f}s"
            )

    # ============ 汇总 ============
    weights = {"cat": 0.48, "lgb": 0.30, "xgb": 0.22}
    present = {k: w for k, w in weights.items() if k in oof}
    s = sum(present.values())
    present = {k: w / s for k, w in present.items()}

    blend_oof = sum(present[k] * oof[k] for k in present)
    blend_test = sum(present[k] * test_pred[k] for k in present)

    final_oof_pred = np.argmax(blend_oof, axis=1)
    cv_score = balanced_accuracy_score(y_num, final_oof_pred)
    total_time = time.time() - t_global_start

    print(f"\n{'=' * 70}")
    print(f"  最终结果")
    print(f"{'=' * 70}")
    for name in model_names:
        single_score = balanced_accuracy_score(y_num, np.argmax(oof[name], axis=1))
        print(
            f"  {MODEL_DISPLAY[name]:>8}  OOF BalAcc: {single_score:.6f}  |  "
            f"权重: {present[name]:.2f}  |  耗时: {model_elapsed[name]:.1f}s"
        )
    print(f"  {'─' * 54}")
    print(f"  Ensemble  OOF BalAcc: {cv_score:.6f}  |  总耗时: {total_time:.1f}s")
    print(f"{'=' * 70}")

    final_pred = np.argmax(blend_test, axis=1)
    pred_labels = [LABEL_LIST[p] for p in final_pred]

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sub = sample_sub.copy()
    sub[TARGET_COL] = pred_labels
    sub.to_csv(save_path, index=False)
    print(f"\n[OK] Submission saved to: {save_path}")


if __name__ == "__main__":
    main()
