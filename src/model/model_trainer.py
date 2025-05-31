import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, roc_auc_score
from src.model.ensemble_models import RandomForestModel, XGBoostModel, LightGBMModel

MODELS_DIR = Path("models")
FEATURES_DIR = Path("data/features")

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def walk_forward_split(df, n_splits, train_period, val_period, date_col="date"):
    """
    Generator for walk-forward splits.
    """
    unique_dates = df.index.get_level_values(0).unique() if isinstance(df.index, pd.MultiIndex) else df[date_col].unique()
    for i in range(n_splits):
        train_start = i * val_period
        train_end = train_start + train_period
        val_start = train_end
        val_end = val_start + val_period
        if val_end > len(unique_dates):
            break
        train_dates = unique_dates[train_start:train_end]
        val_dates = unique_dates[val_start:val_end]
        train_idx = df.index.get_level_values(0).isin(train_dates)
        val_idx = df.index.get_level_values(0).isin(val_dates)
        yield df[train_idx], df[val_idx]

def get_model(model_name, model_type, hyperparameters):
    if model_name.lower() == "randomforest":
        return RandomForestModel(model_type=model_type, hyperparameters=hyperparameters)
    elif model_name.lower() == "xgboost":
        return XGBoostModel(model_type=model_type, hyperparameters=hyperparameters)
    elif model_name.lower() == "lightgbm":
        return LightGBMModel(model_type=model_type, hyperparameters=hyperparameters)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def evaluate_predictions(y_true, y_pred, task="regression"):
    if task == "regression":
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred)
        }
    else:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "auc": roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan
        }

def train_and_validate(
    features_path: str,
    model_config_path: str,
    save_model: bool = True,
    save_preds: bool = True,
    version: str = "v1"
):
    # Load features and config
    df = pd.read_parquet(features_path)
    config = load_config(model_config_path)
    model_name = config["model_name"]
    model_type = config.get("model_type", "regressor")
    hyperparameters = config.get("hyperparameters", {})
    train_period = config.get("train_period", 252*3)
    val_period = config.get("val_period", 21)
    n_splits = config.get("n_splits", 10)
    feature_cols = config.get("feature_cols", [col for col in df.columns if col not in ["target"]])
    target_col = config.get("target_col", "target")
    task = config.get("task", "regression")

    preds_record = []
    metrics_record = []

    for i, (train_df, val_df) in enumerate(walk_forward_split(df, n_splits, train_period, val_period)):
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_val, y_val = val_df[feature_cols], val_df[target_col]

        model = get_model(model_name, model_type, hyperparameters)
        model.train(X_train, y_train)
        y_pred = model.predict(X_val)
        metrics = evaluate_predictions(y_val, y_pred, task=task)
        metrics["split"] = i
        metrics_record.append(metrics)

        val_results = val_df.copy()
        val_results["prediction"] = y_pred
        preds_record.append(val_results[["prediction", target_col]])

        if save_model:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.save(str(MODELS_DIR / f"{model_name}_split{i}_{version}.joblib"))

    # Save predictions and metrics
    if save_preds:
        preds_df = pd.concat(preds_record)
        preds_df.to_parquet(FEATURES_DIR / f"predictions_{model_name}_{version}.parquet")
        pd.DataFrame(metrics_record).to_csv(FEATURES_DIR / f"metrics_{model_name}_{version}.csv", index=False)

    return metrics_record

def main():
    features_path = str(FEATURES_DIR / "features_v1.parquet")
    model_config_path = "config/model_config.yaml"
    train_and_validate(features_path, model_config_path, save_model=True, save_preds=True, version="v1")

if __name__ == "__main__":
    main()