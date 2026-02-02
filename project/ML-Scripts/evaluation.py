import os
import json
import argparse
import tempfile
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, root_mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--preprocessing-run-id", required=True)
    parser.add_argument("--mae-drift-factor", type=float, default=1.2)
    parser.add_argument("--max-mean-residual", type=float, default=5.0)
    ## Dummy
    parser.add_argument("--max-relative-error", type=float, default=10.0)
    parser.add_argument("--p95-relative-error", type=float, default=1.0) 
      
    return parser.parse_args()

def load_best_model_and_data(training_run_id, preprocessing_run_id):
    client = MlflowClient()

    training_run = client.get_run(training_run_id)
    experiment_id = training_run.info.experiment_id
    
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{training_run_id}'",
    )

    best_run = min(
        child_runs,
        key=lambda r: r.data.metrics.get("rmse", float("inf")),
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    tmp_dir = tempfile.mkdtemp()
    client.download_artifacts(
        run_id=preprocessing_run_id,
        path="processed_data",
        dst_path=tmp_dir,
    )

    val_path = os.path.join(tmp_dir, "processed_data", "val.csv")
    val_df = pd.read_csv(val_path)

    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"].values

    return best_run, model, X_val, y_val


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="evaluation") as eval_run:

        run_id = eval_run.info.run_id
        with open("/tmp/run_id.txt", "w") as f:
            f.write(run_id)
    
        client = MlflowClient()
    
        best_run, model, X_val, y_val = load_best_model_and_data(
            args.training_run_id,
            args.preprocessing_run_id
        )

        mlflow.log_param("best_model_run_id", best_run.info.run_id)
        mlflow.log_param("best_model_name", best_run.info.run_name)
        mlflow.log_param("preprocessing_run_id", args.preprocessing_run_id)

        y_pred = model.predict(X_val)
    
        ## Calculatios
        val_mae = mean_absolute_error(y_val, y_pred)
        val_r2 = r2_score(y_val, y_pred)
        val_medae = median_absolute_error(y_val, y_pred)

        val_rmse = root_mean_squared_error(y_val, y_pred)
        std_profit = np.std(y_val)
        std_pred = np.std(y_pred)

        residuals = y_pred - y_val
        abs_errors = np.abs(y_pred - y_val)
             
        train_mae = best_run.data.metrics.get("mae", val_mae)

        ## Gate Constants
        MAE_STD_TRESHOLD_FACTOR = 0.25
        HIT_TOLERANCE_CURRENCY = 25.0
        MIN_HIT_RATE = 0.70
        MIN_VAR_RATIO = 0.7
        MAX_VAR_RATIO = 1.3

        ## Business Gates
        # B1: R2 Score > 0.65
        gate_r2_score = val_r2 > 0.60
        # B2: MAE vs. Std
        gate_mae_vs_std = val_mae < (MAE_STD_TRESHOLD_FACTOR * std_profit)
        # B3: Hit Rate
        hits = np.sum(abs_errors <= HIT_TOLERANCE_CURRENCY)
        current_hit_rate = hits / len(y_val)
        gate_hit_rate = current_hit_rate >= MIN_HIT_RATE

        ## Technical Gates
        # T1: MAE drift
        gate_mae_drift = val_mae <= (train_mae * args.mae_drift_factor)
        # T2: Residual Bias
        gate_residual_bias = abs(residuals.mean()) <= args.max_mean_residual
        # T3: Prediction Variability / Std Ratio
        variability_ratio = std_pred / (std_profit + 1e-6)
        gate_variability = (variability_ratio >= MIN_VAR_RATIO) and (variability_ratio <= MAX_VAR_RATIO)
    
        promote = all([
            gate_r2_score,
            gate_mae_vs_std,
            gate_hit_rate,
            gate_mae_drift,
            gate_residual_bias,
            gate_variability
        ])

        # ---- Deploy Model
        if promote:
            model_uri = f"runs:/{best_run.info.run_id}/model"
            model_name = "sales_profit_model" 
            
            mv = mlflow.register_model(model_uri, model_name)
            
            client = MlflowClient()
            
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Model {model_name} version {mv.version} transitioned to Production.")
    
        # ---- Decision artifact
        decision = {
            "best_model_run_id": best_run.info.run_id,
            "best_model_name": best_run.info.run_name,
            "metrics": {
                "val_mae": float(val_mae),
                "val_r2": float(val_r2),
                "val_rmse": float(val_rmse),
                "data_std": float(std_profit),
                "pred_std": float(std_pred),
                "hit_rate_25": float(current_hit_rate),
                "variability_ratio": float(variability_ratio),
                "max_abs_error": float(abs_errors.max()),
                "mean_residual_bias": float(residuals.mean())
            },
            "gates": {
                "B1_r2_greater_than_0.60": bool(gate_r2_score),
                "B2_mae_smaller_0.25_std": bool(gate_mae_vs_std),
                "B3_hit_rate_gt_70%": bool(gate_hit_rate),
                "T1_mae_drift_passed": bool(gate_mae_drift),
                "T2_residual_bias_passed": bool(gate_residual_bias),
                "T3_variability_passed": bool(gate_variability)
            },
            "promote": promote,
        }
    
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_medae", val_medae)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("max_abs_error", float(abs_errors.max()))
        mlflow.log_metric("mean_residual", float(residuals.mean()))
        mlflow.log_metric("val_hit_rate", current_hit_rate)
        mlflow.log_metric("val_variability_ratio", variability_ratio)

        mlflow.log_param("Gate_B1__r2_greater_than_0.60_passed", gate_r2_score)
        mlflow.log_param("Gate_B2_mae_smaller_than_half_std_passed", gate_mae_vs_std)
        mlflow.log_param("Gate_B3_hit_rate_passed", gate_hit_rate)
        mlflow.log_param("Gate_T1_mae_drift_passed", gate_mae_drift)
        mlflow.log_param("Gate_T2_residual_bias_passed", gate_residual_bias)
        mlflow.log_param("Gate_T3_variability_passed", gate_variability)

        mlflow.log_param("promote", promote)
        
        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/decision.json", "w") as f:
            json.dump(decision, f, indent=2)
    
        mlflow.log_artifact("evaluation/decision.json", artifact_path="evaluation")

        if not promote:
            raise RuntimeError(
                "Model rejected by evaluation gates. See evaluation/decision.json"
            )


if __name__ == "__main__":
    main()
