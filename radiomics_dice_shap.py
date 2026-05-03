import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-validated LightGBM model to predict Dice (or transformed Dice) "
            "from pyradiomics features, with more robust SHAP analysis."
        )
    )
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input CSV with radiomics features and a Dice column.",
    )
    p.add_argument(
        "--target_col",
        type=str,
        default="dice",
        help="Name of the column containing per-case Dice scores.",
    )
    p.add_argument(
        "--target_transform",
        type=str,
        default="none",
        choices=["none", "one_minus", "logit"],
        help="Optional transform applied to the target before regression.",
    )
    p.add_argument(
        "--drop_cols",
        type=str,
        nargs="*",
        default=["label", "filename"],
        help=(
            "Optional extra columns to drop from features "
            "(e.g. classification labels, IDs)."
        ),
    )
    p.add_argument(
        "--min_var",
        type=float,
        default=1e-6,
        help="Drop features with variance lower than this threshold.",
    )
    p.add_argument(
        "--corr_threshold",
        type=float,
        default=0.0,
        help="Drop features whose absolute Pearson correlation with target is "
        "below this value (0 to disable).",
    )
    p.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and model.",
    )
    p.add_argument(
        "--shap_max_samples",
        type=int,
        default=256,
        help="Maximum number of samples used as SHAP background and evaluation subset.",
    )
    p.add_argument(
        "--max_features_for_shap",
        type=int,
        default=None,
        help="If set, only use top-N features (by mean |SHAP|) when plotting.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="radiomics_dice_shap_out",
        help="Directory to save SHAP plots and artifacts.",
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="If set, do not generate SHAP plots (still saves CSV importance).",
    )
    return p.parse_args()


def _load_data(
    csv_path: str,
    target_col: str,
    drop_cols: Optional[List[str]],
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. "
            "Please add it or change --target_col."
        )

    cols = list(df.columns)
    to_exclude = {target_col}
    if drop_cols:
        to_exclude.update(drop_cols)
    feature_cols = [c for c in cols if c not in to_exclude]
    if not feature_cols:
        raise ValueError("No feature columns left after excluding target and drop_cols.")

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float).values
    return X, y


def _summarize_target(y: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    desc = {
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "p25": float(np.percentile(y, 25)),
        "p50": float(np.percentile(y, 50)),
        "p75": float(np.percentile(y, 75)),
    }
    print("Target summary:", desc)
    pd.Series(desc).to_json(os.path.join(out_dir, "target_summary.json"), indent=2)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(6, 4))
        plt.hist(y, bins=30, edgecolor="black")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.title("Target distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "target_hist.png"), dpi=200)
        plt.close()
    except Exception:
        # plotting is optional
        pass


def _transform_target(
    y: np.ndarray, mode: str
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    if mode == "none":
        return y, lambda x: x
    if mode == "one_minus":
        return 1.0 - y, lambda x: 1.0 - x
    if mode == "logit":
        eps = 1e-6
        y_clip = np.clip(y, eps, 1.0 - eps)
        z = np.log(y_clip / (1.0 - y_clip))
        return z, lambda x: 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unknown target_transform mode: {mode}")


def _filter_features(
    X: pd.DataFrame,
    y: np.ndarray,
    min_var: float,
    corr_threshold: float,
) -> pd.DataFrame:
    n_before = X.shape[1]

    # 1) variance filter
    var = X.var(axis=0)
    keep = var > min_var
    X_f = X.loc[:, keep]

    # 2) correlation filter
    if corr_threshold > 0.0 and X_f.shape[1] > 0:
        corrs = []
        for col in X_f.columns:
            try:
                c = np.corrcoef(X_f[col].values, y)[0, 1]
            except Exception:
                c = 0.0
            if not np.isfinite(c):
                c = 0.0
            corrs.append(c)
        corrs = np.asarray(corrs)
        keep_corr = np.abs(corrs) >= corr_threshold
        X_f = X_f.loc[:, keep_corr]

    n_after = X_f.shape[1]
    print(f"Feature filtering: {n_before} -> {n_after} columns kept.")
    if n_after == 0:
        raise ValueError("All features were filtered out; relax min_var/corr_threshold.")
    return X_f


def _train_lightgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_data_in_leaf=20,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _train_cv_models(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int,
):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = []
    metrics_list: List[Dict[str, float]] = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr = X.iloc[tr_idx].values
        y_tr = y[tr_idx]
        X_val = X.iloc[val_idx].values
        y_val = y[val_idx]

        model = _train_lightgbm_regressor(X_tr, y_tr, seed=seed + fold)
        y_pred = model.predict(X_val)
        metrics = _eval_regression(y_val, y_pred)
        metrics_list.append(metrics)

        print(
            f"Fold {fold}: MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
        )
        models.append(model)

    # aggregate metrics
    agg = {
        "mae_mean": float(np.mean([m["mae"] for m in metrics_list])),
        "mae_std": float(np.std([m["mae"] for m in metrics_list])),
        "rmse_mean": float(np.mean([m["rmse"] for m in metrics_list])),
        "rmse_std": float(np.std([m["rmse"] for m in metrics_list])),
        "r2_mean": float(np.mean([m["r2"] for m in metrics_list])),
        "r2_std": float(np.std([m["r2"] for m in metrics_list])),
    }
    print(
        "CV summary: "
        f"MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}, "
        f"RMSE={agg['rmse_mean']:.4f}±{agg['rmse_std']:.4f}, "
        f"R2={agg['r2_mean']:.4f}±{agg['r2_std']:.4f}"
    )
    return models, agg


def _run_shap_cv(
    models: List[object],
    X: pd.DataFrame,
    shap_max_samples: int,
    out_dir: str,
    max_features_for_shap: Optional[int],
    no_plots: bool,
) -> None:
    import shap  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    os.makedirs(out_dir, exist_ok=True)

    if shap_max_samples is not None and shap_max_samples > 0 and len(X) > shap_max_samples:
        X_sample = X.sample(n=shap_max_samples, random_state=0)
    else:
        X_sample = X

    all_shap = []
    for m in models:
        explainer = shap.TreeExplainer(m)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # For regression TreeExplainer normally returns array, but handle list just in case.
            shap_values = shap_values[0]
        all_shap.append(shap_values)

    all_shap_arr = np.stack(all_shap, axis=0)  # (n_models, n_samples, n_features)
    shap_mean = all_shap_arr.mean(axis=0)  # (n_samples, n_features)

    mean_abs = np.abs(shap_mean).mean(axis=0)
    feature_importance = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)
    importance_path = os.path.join(out_dir, "shap_global_importance.csv")
    feature_importance.to_csv(importance_path, header=["mean_abs_shap"])
    print(f"Global SHAP importance saved to: {importance_path}")

    if no_plots:
        return

    # Optionally restrict to top-N features for plotting
    if max_features_for_shap is not None and max_features_for_shap > 0:
        top_feats = feature_importance.head(max_features_for_shap).index
        X_plot = X_sample[top_feats]
        idxs = [X_sample.columns.get_loc(c) for c in top_feats]
        shap_plot = shap_mean[:, idxs]
    else:
        X_plot = X_sample
        shap_plot = shap_mean

    # Summary beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_plot, X_plot, show=False)
    summary_path = os.path.join(out_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_plot, X_plot, plot_type="bar", show=False)
    bar_path = os.path.join(out_dir, "shap_importance_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    print(f"SHAP summary plots saved to: {summary_path}, {bar_path}")


def main() -> None:
    args = _parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_raw, y_raw = _load_data(args.csv, args.target_col, args.drop_cols)
    print(f"Loaded {len(X_raw)} samples with {X_raw.shape[1]} features.")

    _summarize_target(y_raw, args.out_dir)

    y_trans, inverse_fn = _transform_target(y_raw, args.target_transform)
    print(f"Applied target_transform='{args.target_transform}'.")

    X = _filter_features(X_raw, y_trans, args.min_var, args.corr_threshold)
    print(f"Training with {X.shape[1]} filtered features.")

    models, cv_metrics = _train_cv_models(X, y_trans, args.n_splits, args.seed)

    if cv_metrics["r2_mean"] < 0:
        print(
            "Warning: mean CV R2 < 0; the model does not outperform a constant baseline. "
            "SHAP explanations may have limited interpretability."
        )

    print("Running SHAP analysis (TreeExplainer on CV ensemble)...")
    _run_shap_cv(
        models=models,
        X=X,
        shap_max_samples=args.shap_max_samples,
        out_dir=args.out_dir,
        max_features_for_shap=args.max_features_for_shap,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()


