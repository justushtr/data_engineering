import os
import time
import re
import argparse
import subprocess
import pandas as pd
import boto3
from botocore.client import Config
from io import BytesIO

def get_s3_resource():
    return boto3.resource(
        's3',
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", default="input-bucket")
    parser.add_argument("--prep-bucket", default="preparation-data-bucket")
    parser.add_argument("--model-bucket", default="data-for-model-bucket")
    parser.add_argument("--experiment-name", default="automotive-training-cycle")
    parser.add_argument("--poll-interval", type=int, default=10)
    return parser.parse_args()

def get_latest_master_file(bucket_obj, prefix="all_data_"):
    """Findet die Datei mit der höchsten Version (z.B. all_data_5.csv)."""
    max_ver = 0
    latest_file = None
    pattern = re.compile(rf"{prefix}(\d+)\.csv")
    
    for obj in bucket_obj.objects.all():
        match = pattern.match(obj.key)
        if match:
            version = int(match.group(1))
            if version > max_ver:
                max_ver = version
                latest_file = obj.key
    return latest_file, max_ver

def run_step(script_name, args_list):
    """Hilfsfunktion, um deine anderen Python-Skripte als Subprozess zu starten."""
    cmd = ["python", script_name] + args_list
    print(f"🚀 Starte {script_name}...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Fehler in {script_name}:\n{result.stderr}")
        raise RuntimeError(f"{script_name} fehlgeschlagen.")
    
    print(f"✅ {script_name} abgeschlossen.")
    # Optional: Logs ausgeben
    # print(result.stdout)

def read_run_id(filepath):
    """Liest die Run-ID, die deine Skripte in /tmp speichern."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
    return None

def main():
    args = parse_args()
    s3 = get_s3_resource()
    input_bucket = s3.Bucket(args.input_bucket)
    
    print(f"👀 Watcher gestartet. Überwache '{args.input_bucket}'...")

    while True:
        try:
            # 1. Nach neuen CSVs im Input-Bucket suchen
            found_files = [obj.key for obj in input_bucket.objects.all() if obj.key.endswith(".csv")]
            
            for new_filename in found_files:
                print(f"\n--- ⚡ Neue Datei gefunden: {new_filename} ---")
                
                # A. Daten laden & Mergen
                # -----------------------
                print("Verarbeite Daten...")
                # Neue Datei laden
                obj_new = input_bucket.Object(new_filename).get()
                df_new = pd.read_csv(BytesIO(obj_new['Body'].read()))
                
                # Master Datei laden
                prep_bucket = s3.Bucket(args.prep_bucket)
                latest_master, current_ver = get_latest_master_file(prep_bucket)
                
                if latest_master:
                    print(f"Merge mit Master: {latest_master}")
                    obj_master = prep_bucket.Object(latest_master).get()
                    df_master = pd.read_csv(BytesIO(obj_master['Body'].read()))
                    df_combined = pd.concat([df_master, df_new], ignore_index=True)
                else:
                    print("Erstelle neue Master-Datei.")
                    df_combined = df_new

                # Neue Version speichern
                new_ver = current_ver + 1
                new_master_name = f"all_data_{new_ver}.csv"
                
                csv_buffer = BytesIO()
                df_combined.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                # Upload in Prep Bucket (Archiv)
                prep_bucket.put_object(Key=new_master_name, Body=csv_buffer.getvalue())
                
                # Upload in Model Bucket (Für das aktuelle Training)
                # Wir nennen sie hier dynamisch, wie im Argument verlangt
                model_bucket = s3.Bucket(args.model_bucket)
                model_bucket.put_object(Key=new_master_name, Body=csv_buffer.getvalue())
                
                print(f"Daten gespeichert als: {new_master_name}")

                # B. Die ML-Pipeline starten
                # --------------------------
                
                # Schritt 1: EDA
                run_step("exploratory_data_analysis.py", [
                    "--experiment-name-mlflow", args.experiment_name,
                    "--bucket-name", args.model_bucket,
                    "--filename", new_master_name
                ])

                # Schritt 2: Preprocessing
                # Preprocessing speichert Run-ID in /tmp/preprocessing_run_id.txt
                run_step("preprocessing.py", [
                    "--experiment-name-mlflow", args.experiment_name,
                    "--bucket-name", args.model_bucket,
                    "--filename", new_master_name,
                    "--output-path", "processed_data" # Lokal temporär
                ])
                preproc_run_id = read_run_id("/tmp/preprocessing_run_id.txt")

                # Schritt 3: Training
                # Benötigt die ID vom Preprocessing
                run_step("train.py", [
                    "--experiment-name-mlflow", args.experiment_name,
                    "--preprocessing-run-id", preproc_run_id
                ])
                train_run_id = read_run_id("/tmp/train_run_id.txt")

                # Schritt 4: Evaluation
                # Benötigt ID von Training UND Preprocessing
                run_step("evaluation.py", [
                    "--experiment-name-mlflow", args.experiment_name,
                    "--training-run-id", train_run_id,
                    "--preprocessing-run-id", preproc_run_id
                ])

                print("✅ Pipeline-Zyklus erfolgreich beendet.")

                # C. Cleanup
                # ----------
                print(f"Lösche {new_filename} aus Input...")
                input_bucket.Object(new_filename).delete()
            
            # Kurz schlafen um CPU zu sparen
            time.sleep(args.poll_interval)

        except Exception as e:
            print(f"❌ Kritischer Fehler im Watcher-Loop: {e}")
            time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()