import os
import time
import re
import argparse
import pandas as pd
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from io import BytesIO

# --- KONFIGURATION & SETUP ---

def get_s3_resource():
    """Verbindung zu MinIO/S3 herstellen."""
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
    parser.add_argument("--poll-interval", type=int, default=10)
    return parser.parse_args()

def ensure_infrastructure(s3_resource, buckets):
    """Erstellt Buckets, falls sie noch nicht existieren."""
    print("--- 🏗️ Prüfe Infrastruktur (Buckets) ---")
    for bucket_name in buckets:
        try:
            s3_resource.create_bucket(Bucket=bucket_name)
            print(f"✅ Bucket bereit: {bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['BucketAlreadyOwnedByYou', 'BucketAlreadyExists']:
                print(f"🆗 Bucket existiert bereits: {bucket_name}")
            else:
                print(f"❌ Fehler beim Erstellen von {bucket_name}: {e}")
                raise e

def get_latest_master_file(bucket_obj, prefix="all_data_"):
    """
    Findet die Datei mit der höchsten Version (z.B. all_data_5.csv).
    Gibt (Dateiname, Version) zurück. Wenn nichts gefunden, (None, 0).
    """
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

# --- HAUPTPROGRAMM ---

def main():
    args = parse_args()
    s3 = get_s3_resource()
    
    # 1. Infrastruktur-Check
    required_buckets = [args.input_bucket, args.prep_bucket, args.model_bucket]
    ensure_infrastructure(s3, required_buckets)

    input_bucket = s3.Bucket(args.input_bucket)
    prep_bucket = s3.Bucket(args.prep_bucket)
    model_bucket = s3.Bucket(args.model_bucket)
    
    print(f"👀 Watcher gestartet. Warte auf Dateien in '{args.input_bucket}'...")

    while True:
        try:
            # Suche nach .csv Dateien im Input
            found_files = [obj.key for obj in input_bucket.objects.all() if obj.key.endswith(".csv")]
            
            for new_filename in found_files:
                print(f"\n⚡ Neue Datei entdeckt: {new_filename}")
                
                # --- SCHRITT 1: Daten laden & zusammenfügen ---
                
                # A. Neue Datei laden
                obj_new = input_bucket.Object(new_filename).get()
                df_new = pd.read_csv(BytesIO(obj_new['Body'].read()))
                
                # B. Master-Datei suchen & laden
                latest_master, current_ver = get_latest_master_file(prep_bucket)
                
                if latest_master:
                    print(f"⏳ Lade aktuelle Master-Datei: {latest_master}")
                    obj_master = prep_bucket.Object(latest_master).get()
                    df_master = pd.read_csv(BytesIO(obj_master['Body'].read()))
                    
                    # Anhängen
                    df_combined = pd.concat([df_master, df_new], ignore_index=True)
                else:
                    print("🆕 Kaltstart: Erstelle erste Version (all_data_1.csv).")
                    df_combined = df_new
                    current_ver = 0

                # --- SCHRITT 2: Speichern & Pipeline triggern ---
                
                new_ver = current_ver + 1
                new_master_name = f"all_data_{new_ver}.csv"
                
                csv_buffer = BytesIO()
                df_combined.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue()
                
                # Upload 1: In den Preparation Bucket (Historie/Sicherung)
                prep_bucket.put_object(Key=new_master_name, Body=csv_bytes)
                
                # Upload 2: In den Model Bucket (Das ist der TRIGGER für deine YAML-Pipeline)
                print(f"🚀 Lade {new_master_name} in Model-Bucket hoch (Trigger)...")
                model_bucket.put_object(Key=new_master_name, Body=csv_bytes)
                
                print(f"✅ Erfolg: {new_master_name} bereitgestellt.")

                # --- SCHRITT 3: Aufräumen ---
                print(f"🗑️ Lösche Input-Datei {new_filename}...")
                input_bucket.Object(new_filename).delete()
            
            time.sleep(args.poll_interval)

        except Exception as e:
            print(f"❌ Fehler: {e}")
            time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()