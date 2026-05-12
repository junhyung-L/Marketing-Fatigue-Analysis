import os
import json
import argparse
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

from data_loader import load_and_merge_data, stratified_fixed_sample, split_random_row_simple, prepare_data_2d, prepare_data_3d
from metrics import compute_classification_metrics
import models

def parse_args():
    parser = argparse.ArgumentParser(description="Train models for ecommerce journey project.")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "rf", "mlp", "cnn", "lstm", "cnnlstm", "rnn", "rnnlstm"], help="Model type to train")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save models and metrics")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load and Split Data
    raw_data = load_and_merge_data(data_dir=args.data_dir)
    
    # 100k Sampling
    dataset = stratified_fixed_sample(raw_data, target_col="is_purchased", total_n=100_000, pos_ratio=0.10, seed=1)
    
    # Split
    train_df, val_df, test_df = split_random_row_simple(dataset, val_frac=0.15, test_frac=0.15, seed=1)
    
    # Feature Selection (v0 and v1 for simplicity in this example script)
    # In full implementation, should use the specific features used in notebooks
    feature_columns = sorted([c for c in dataset.columns if c != "is_purchased" and c != "id"])
    
    print(f"Using {len(feature_columns)} features.")

    # 2. Prepare Data Shapes and Train
    if args.model in ["xgb", "rf", "mlp"]:
        Xtr, Ytr = prepare_data_2d(train_df, feature_columns)
        Xva, Yva = prepare_data_2d(val_df, feature_columns)
        Xte, Yte = prepare_data_2d(test_df, feature_columns)
    else:
        Xtr, Ytr = prepare_data_3d(train_df, feature_columns)
        Xva, Yva = prepare_data_3d(val_df, feature_columns)
        Xte, Yte = prepare_data_3d(test_df, feature_columns)

    print(f"Train Shape: {Xtr.shape}")

    # 3. Model Training
    if args.model == "xgb":
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=1, device="cuda" if tf.config.list_physical_devices('GPU') else "cpu")
        model.fit(Xtr, Ytr, eval_set=[(Xva, Yva)], verbose=10)
        y_prob = model.predict_proba(Xte)[:, 1]
        
    elif args.model == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
        model.fit(Xtr, Ytr)
        y_prob = model.predict_proba(Xte)[:, 1]
        
    elif args.model == "mlp":
        model = models.create_mlp(input_shape=(Xtr.shape[1],))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=10, batch_size=32)
        y_prob = model.predict(Xte).flatten()
        
    elif args.model == "cnn":
        model = models.create_cnn(input_shape=(Xtr.shape[1], Xtr.shape[2]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=10, batch_size=32)
        y_prob = model.predict(Xte).flatten()
        
    elif args.model == "lstm":
        model = models.create_lstm(input_shape=(Xtr.shape[1], Xtr.shape[2]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=10, batch_size=32)
        y_prob = model.predict(Xte).flatten()
        
    elif args.model == "cnnlstm":
        model = models.create_cnnlstm(input_shape=(Xtr.shape[1], Xtr.shape[2]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=10, batch_size=32)
        y_prob = model.predict(Xte).flatten()
        
    elif args.model == "rnn":
        model = models.create_rnn(input_shape=(Xtr.shape[1], Xtr.shape[2]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=10, batch_size=32)
        y_prob = model.predict(Xte).flatten()
        
    elif args.model == "rnnlstm":
        model = models.create_rnnlstm(input_shape=(Xtr.shape[1], Xtr.shape[2]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=10, batch_size=32)
        y_prob = model.predict(Xte).flatten()

    # 4. Evaluation
    metrics, cm = compute_classification_metrics(Yte, y_prob, thr=0.5, n_features=Xtr.shape[1])
    
    print("\n[Final Metrics]")
    print(json.dumps(metrics, indent=2))
    print(f"Confusion Matrix: {cm}")

    # Save Metrics
    output_path = os.path.join(args.output_dir, f"{args.model}_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {output_path}")

if __name__ == "__main__":
    main()
