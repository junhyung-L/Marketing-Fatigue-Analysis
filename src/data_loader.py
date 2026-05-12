import os
import gc
import numpy as np
import pandas as pd

def load_and_merge_data(data_dir=".", versions=["v1", "v2", "v3", "v4"]):
    """
    Loads base feature v0 and merges with v1~v4 features.
    """
    path_v0 = os.path.join(data_dir, "featured_v0.parquet")
    print(f"[LOAD] Base: {path_v0}")
    final_data = pd.read_parquet(path_v0)
    print(f"   -> Base Shape: {final_data.shape}")

    for ver in versions:
        path_ver = os.path.join(data_dir, f"featured_{ver}.parquet")
        if os.path.exists(path_ver):
            print(f"[MERGE] Checking {ver}...")
            temp_df = pd.read_parquet(path_ver)
            
            # Extract new columns only
            new_cols = [c for c in temp_df.columns if c not in final_data.columns]
            
            if new_cols:
                # Concatenate by index (assuming order is identical)
                final_data = pd.concat([final_data, temp_df[new_cols]], axis=1)
                print(f"   -> Added {len(new_cols)} features from {ver}")
            else:
                print(f"   -> No new features in {ver}")
                
            del temp_df
            gc.collect()
        else:
            print(f"[WARN] File not found: {path_ver}")

    print(f"\n[FINAL DATA] Total Shape: {final_data.shape}")
    return final_data

def stratified_fixed_sample(df, target_col="is_purchased", total_n=100_000, pos_ratio=0.10, seed=1):
    """
    Performs stratified sampling to get fixed size and positive ratio.
    """
    n_pos_k = int(round(total_n * pos_ratio))
    n_neg_k = total_n - n_pos_k
    
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    
    pos_s = pos.sample(n=min(n_pos_k, len(pos)), random_state=seed, replace=False)
    neg_s = neg.sample(n=min(n_neg_k, len(neg)), random_state=seed, replace=False)
    
    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def split_random_row_simple(df, val_frac=0.15, test_frac=0.15, seed=1):
    """
    Splits data randomly into train, val, and test.
    """
    rng_local = np.random.RandomState(seed)
    N = len(df)
    idx = np.arange(N)
    rng_local.shuffle(idx)
    
    n_test = int(round(N * test_frac))
    n_val  = int(round(N * val_frac))
    
    te_idx = idx[:n_test]
    va_idx = idx[n_test:n_test+n_val]
    tr_idx = idx[n_test+n_val:]
    
    te = df.iloc[te_idx].reset_index(drop=True)
    va = df.iloc[va_idx].reset_index(drop=True)
    tr = df.iloc[tr_idx].reset_index(drop=True)
    
    return tr, va, te

def prepare_data_2d(df, feature_columns, target_col="is_purchased"):
    """
    Prepares 2D data for ML models (N, F).
    """
    X = df[feature_columns].fillna(0).astype(np.float32).to_numpy()
    Y = df[target_col].fillna(0).astype(np.int32).to_numpy()
    return X, Y

def prepare_data_3d(df, feature_columns, target_col="is_purchased"):
    """
    Prepares 3D data for DL models (N, F, 1).
    """
    X = df[feature_columns].fillna(0).astype(np.float32).to_numpy()
    X = X.reshape((X.shape[0], X.shape[1], 1))
    Y = df[target_col].fillna(0).astype(np.int32).to_numpy()
    return X, Y
