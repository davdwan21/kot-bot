import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config
@dataclass
class Config:
    data_root: str = "data"
    csv_glob: str = "**/train.csv"

    # Label + drop columns
    label_col: str = "jump_next"
    drop_cols: tuple = ("frame_idx", "t")  # keep dt

    # Training
    batch_size: int = 1024
    epochs: int = 15
    lr: float = 2e-3
    weight_decay: float = 1e-4
    val_frac_runs: float = 0.15
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    out_path: str = "baseline_mlp_jump.pt"

cfg = Config()

# Util methods
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(precision), float(recall), float(f1)

def load_all_runs(data_root: str, csv_glob: str):
    pattern = os.path.join(data_root, csv_glob)
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No CSVs found. Looked for: {pattern}\n"
            f"Tip: set cfg.data_root to your dataset root folder."
        )

    dfs = []
    for run_id, p in enumerate(paths):
        df = pd.read_csv(p)

        # Attach run metadata (not used as features)
        df["_run_id"] = run_id
        df["_run_path"] = p
        dfs.append(df)

    return dfs, paths


# Dataset
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
# Model
class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
def main():
    set_seed(cfg.seed)

    dfs, paths = load_all_runs(cfg.data_root, cfg.csv_glob)
    n_runs = len(dfs)

    # ---- split by run ----
    rng = np.random.default_rng(cfg.seed)
    run_ids = np.arange(n_runs)
    rng.shuffle(run_ids)

    n_val_runs = max(1, int(round(cfg.val_frac_runs * n_runs)))
    val_runs = set(run_ids[:n_val_runs])
    tr_runs = set(run_ids[n_val_runs:])

    df_tr = pd.concat([dfs[i] for i in tr_runs], ignore_index=True)
    df_va = pd.concat([dfs[i] for i in val_runs], ignore_index=True)

    # Basic label checks
    if cfg.label_col not in df_tr.columns:
        raise KeyError(f"Label col '{cfg.label_col}' not found in CSV columns.")
    if cfg.label_col not in df_va.columns:
        raise KeyError(f"Label col '{cfg.label_col}' not found in CSV columns.")

    # Ensure labels are 0/1
    y_tr = df_tr[cfg.label_col].astype(int).to_numpy()
    y_va = df_va[cfg.label_col].astype(int).to_numpy()

    # Build feature columns from train columns (consistent ordering!)
    drop = set(cfg.drop_cols) | {cfg.label_col, "_run_id", "_run_path"}
    feat_cols = [c for c in df_tr.columns if c not in drop]

    # Ensure val contains same columns
    missing_in_val = [c for c in feat_cols if c not in df_va.columns]
    if missing_in_val:
        raise KeyError(f"Validation is missing columns: {missing_in_val[:10]} ...")

    # Coerce to float
    X_tr = df_tr[feat_cols].astype(float).to_numpy()
    X_va = df_va[feat_cols].astype(float).to_numpy()

    # Standardize using train stats only
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-6] = 1.0

    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    train_ds = TabDataset(X_tr, y_tr)
    val_ds = TabDataset(X_va, y_va)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = MLP(d_in=X_tr.shape[1]).to(cfg.device)
    
    # Handle class imbalance (# negatives >> # positives)
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - y_tr.sum())
    pos_weight_val = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_val], device=cfg.device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Logging
    pos_rate_tr = (y_tr.mean() if len(y_tr) else 0.0)
    pos_rate_va = (y_va.mean() if len(y_va) else 0.0)

    print(f"device={cfg.device}")
    print(f"found runs={n_runs}  (train runs={len(tr_runs)} val runs={len(val_runs)})")
    print(f"rows train={len(df_tr):,} val={len(df_va):,}")
    print(f"pos_rate train={pos_rate_tr:.4f} val={pos_rate_va:.4f}  pos_weight={pos_weight_val:.2f}")
    print(f"features={len(feat_cols)}  example={feat_cols[:10]} ...")

    best_f1 = -1.0
    best_state = None
    
    for ep in range(1, cfg.epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(len(train_ds), 1)
        
        # Validate
        model.eval()
        va_loss = 0.0
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits = model(xb)
                loss = criterion(logits, yb)

                va_loss += loss.item() * xb.size(0)
                all_logits.append(logits.detach().cpu().numpy())
                all_y.append(yb.detach().cpu().numpy())

        va_loss /= max(len(val_ds), 1)
        logits_np = np.concatenate(all_logits) if all_logits else np.zeros((0,), dtype=np.float32)
        y_np = np.concatenate(all_y).astype(int) if all_y else np.zeros((0,), dtype=np.int32)
        
        # Threshold to maximize F1 on val
        probs = sigmoid(logits_np) if len(logits_np) else np.zeros((0,), dtype=np.float32)
        best_thr, best_ep_f1 = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            pred = (probs >= thr).astype(int)
            _, _, f1 = precision_recall_f1(y_np, pred)
            if f1 > best_ep_f1:
                best_ep_f1 = f1
                best_thr = float(thr)

        pred = (probs >= best_thr).astype(int)
        p, r, f1 = precision_recall_f1(y_np, pred)
        
        print(f"ep {ep:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
              f"| thr {best_thr:.2f} | P {p:.3f} R {r:.3f} F1 {f1:.3f}")

        print(
            f"ep {ep:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
            f"| thr {best_thr:.2f} | P {p:.3f} R {r:.3f} F1 {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                "model": model.state_dict(),
                "mu": mu,
                "sd": sd,
                "feat_cols": feat_cols,
                "thr": best_thr,
                "cfg": cfg.__dict__,
                "val_f1": best_f1,
            }

    if best_state is not None:
        torch.save(best_state, cfg.out_path)
        print(f"saved best checkpoint: {cfg.out_path} (best_val_f1={best_f1:.3f})")
    else:
        print("No checkpoint saved (empty dataset?)")


if __name__ == "__main__":
    main()