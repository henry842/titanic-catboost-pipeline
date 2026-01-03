"""
Titanic (rule-compliant, leaderboard-ready):
- Strong, leak-safe features (incl. K-fold Target Mean Encoding with smoothing)
- CatBoost main model with early stopping & quiet logs
- OOF threshold optimization for Accuracy
- Optional single-round pseudo-labeling (safe version)
"""

import os, re, warnings, random
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

RANDOM_STATE = 42
N_SPLITS = 5
TE_SMOOTHING = 20.0
EARLY_STOPPING_ROUNDS = 200
PSEUDO_LABEL = True          # set to False if you don't want pseudo-labeling
PL_POS_THR, PL_NEG_THR = 0.98, 0.02

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def detect_input_dir():
    for p in ["../input/titanic", "../input/Titanic", "./"]:
        if os.path.exists(os.path.join(p, "train.csv")): return p
    return "./"

def extract_title(name: str) -> str:
    if pd.isna(name): return "Unknown"
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "Unknown"

def clean_ticket_prefix(s: str) -> str:
    if pd.isna(s): return "NONE"
    s = str(s).upper()
    s = re.sub(r"[./]", "", s); s = re.sub(r"\s+", "", s); s = re.sub(r"\d+", "", s)
    return s if s else "NONE"

def preprocess_full(df: pd.DataFrame, train_len: int) -> pd.DataFrame:
    """Feature engineering using only official data; rare-title mapping decided on TRAIN ONLY."""
    out = df.copy()

    # Normalize text columns
    out["Sex"] = out["Sex"].astype(str).str.lower()
    out["Embarked"] = out["Embarked"].astype(str).str.upper()

    # Name-based
    out["Title"] = out["Name"].apply(extract_title)
    # Rare mapping decided from TRAIN ONLY (no label leakage, but avoids test-driven categories)
    title_counts_train = out.iloc[:train_len]["Title"].value_counts()
    rare_titles = title_counts_train[title_counts_train < 10].index
    out["Title"] = out["Title"].where(~out["Title"].isin(rare_titles), "Rare")
    out["LastName"] = out["Name"].fillna("").apply(lambda s: s.split(",")[0].strip() if "," in s else "Unknown")

    # Family / group
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Ticket
    out["TicketPrefix"] = out["Ticket"].apply(clean_ticket_prefix)
    ticket_counts = out["Ticket"].fillna("NA").value_counts()
    out["TicketGroupSize"] = out["Ticket"].fillna("NA").map(ticket_counts).fillna(1).astype(int)

    # Cabin
    out["HasCabin"] = (~out["Cabin"].isna()).astype(int)
    out["CabinDeck"] = out["Cabin"].fillna("M").astype(str).str[0].str.upper()
    out["CabinMulti"] = out["Cabin"].astype(str).str.contains(r"\s").astype(int)

    # Embarked valid set + fill mode (computed on TRAIN ONLY)
    out["Embarked"] = out["Embarked"].where(out["Embarked"].isin(["S", "C", "Q"]), np.nan)
    mode_val = out.iloc[:train_len]["Embarked"].mode()
    mode_val = mode_val.iloc[0] if not mode_val.empty else "S"
    out["Embarked"] = out["Embarked"].fillna(mode_val)

    # Numerics
    out["Pclass"] = out["Pclass"].astype(int)
    out["Fare"] = out["Fare"].astype(float)
    out["Age"] = out["Age"].astype(float)

    # Impute Fare by Pclass (TRAIN statistics applied to ALL)
    pclass_med = out.iloc[:train_len].groupby("Pclass")["Fare"].median()
    out["Fare"] = out.apply(lambda r: pclass_med[r["Pclass"]] if pd.isna(r["Fare"]) else r["Fare"], axis=1)

    # Impute Age by (Title,Pclass,Sex) median (TRAIN statistics)
    grp_med = out.iloc[:train_len].groupby(["Title","Pclass","Sex"])["Age"].median()
    def impute_age(r):
        if not pd.isna(r["Age"]): return r["Age"]
        key = (r["Title"], r["Pclass"], r["Sex"])
        return grp_med.get(key, out.iloc[:train_len]["Age"].median())
    out["Age"] = out.apply(impute_age, axis=1)

    # Derived
    out["TicketGroupSize"] = out["TicketGroupSize"].clip(lower=1)
    out["FarePerPerson"] = out["Fare"] / out["TicketGroupSize"]
    out["IsChild"] = (out["Age"] < 16).astype(int)
    out["IsMother"] = ((out["Sex"] == "female") & (out["Parch"] > 0) & (out["Title"] == "Mrs")).astype(int)

    # Gentle winsorization
    out["Fare"] = out["Fare"].clip(lower=0, upper=np.nanpercentile(out["Fare"], 99))
    out["FarePerPerson"] = out["FarePerPerson"].clip(lower=0, upper=np.nanpercentile(out["FarePerPerson"], 99))
    out["Age"] = out["Age"].clip(lower=0, upper=80)

    return out

def kfold_target_mean_encode(X_tr: pd.DataFrame, y_tr: pd.Series,
                             X_te: pd.DataFrame, col: str,
                             n_splits=5, smoothing=20.0, random_state=42) -> Tuple[np.ndarray, np.ndarray]:
    """Leak-safe K-fold target mean encoding with smoothing."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prior = float(y_tr.mean())

    X_tr_col = X_tr[col].reset_index(drop=True)
    y_tr_ser = pd.Series(y_tr).reset_index(drop=True)
    X_te_col = X_te[col].reset_index(drop=True)

    oof = np.zeros(len(X_tr_col), dtype=float)
    for tr_idx, va_idx in skf.split(np.zeros(len(y_tr_ser)), y_tr_ser):
        df_fold = pd.DataFrame({col: X_tr_col.iloc[tr_idx].values, "y": y_tr_ser.iloc[tr_idx].values})
        stats = df_fold.groupby(col)["y"].agg(["mean","count"])
        smooth = (stats["mean"]*stats["count"] + prior*smoothing) / (stats["count"] + smoothing)
        oof[va_idx] = X_tr_col.iloc[va_idx].map(smooth).fillna(prior).values

    df_full = pd.DataFrame({col: X_tr_col.values, "y": y_tr_ser.values})
    stats_full = df_full.groupby(col)["y"].agg(["mean","count"])
    smooth_full = (stats_full["mean"]*stats_full["count"] + prior*smoothing) / (stats_full["count"] + smoothing)
    te = X_te_col.map(smooth_full).fillna(prior).values
    return oof, te

# ============= Main pipeline =============
seed_everything(RANDOM_STATE)
INPUT_DIR = detect_input_dir()

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

y = train["Survived"].astype(int)
train_nolabel = train.drop(columns=["Survived"])
full = pd.concat([train_nolabel, test], axis=0, ignore_index=True)

full = preprocess_full(full, train_len=len(train))

base_cat = ["Sex","Embarked","Title","CabinDeck","TicketPrefix"]
base_num = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone",
            "IsChild","IsMother","HasCabin","CabinMulti","TicketGroupSize","FarePerPerson"]

# K-fold TE on high-cardinality groups
te_cols = ["LastName","TicketPrefix","CabinDeck","Title"]
n_train = len(train)
train_idx = np.arange(n_train)
test_idx  = np.arange(n_train, len(full))

X_all = full.copy()
te_train_parts, te_test_parts = {}, {}
for c in te_cols:
    oof_c, test_c = kfold_target_mean_encode(
        X_all.loc[train_idx, [c]], y, X_all.loc[test_idx, [c]],
        col=c, n_splits=N_SPLITS, smoothing=TE_SMOOTHING, random_state=RANDOM_STATE
    )
    te_train_parts[c+"_TE"] = oof_c
    te_test_parts[c+"_TE"]  = test_c

for newc, arr in te_train_parts.items():
    X_all.loc[train_idx, newc] = arr
for newc, arr in te_test_parts.items():
    X_all.loc[test_idx, newc] = arr

use_cols = base_cat + base_num + [c+"_TE" for c in te_cols]
X_train = X_all.loc[train_idx, use_cols].reset_index(drop=True)
X_test  = X_all.loc[test_idx,  use_cols].reset_index(drop=True)

# ---- CatBoost training (CV) ----
from catboost import CatBoostClassifier, Pool
cat_idx = [X_train.columns.get_loc(c) for c in base_cat]  # TE are numeric

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(X_train), dtype=float)
test_pred_folds = np.zeros((N_SPLITS, len(X_test)), dtype=float)
fold_acc = []

for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
    X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    est = CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=3000, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="Accuracy",
        random_seed=RANDOM_STATE+fold, od_type="Iter", od_wait=EARLY_STOPPING_ROUNDS, verbose=False
    )
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    va_pool = Pool(X_va, y_va, cat_features=cat_idx)
    te_pool = Pool(X_test, cat_features=cat_idx)

    est.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    va_p = est.predict_proba(va_pool)[:,1]
    te_p = est.predict_proba(te_pool)[:,1]

    oof[va] = va_p
    test_pred_folds[fold-1,:] = te_p
    acc = accuracy_score(y_va, (va_p>=0.5).astype(int))
    fold_acc.append(acc)

print("CatBoost CV Acc per fold:", [f"{a:.4f}" for a in fold_acc], "| Mean:", f"{np.mean(fold_acc):.4f}")

# ---- OOF threshold tuning ----
ths = np.linspace(0.3, 0.7, 401)
best_t, best_acc = max(((t, accuracy_score(y, (oof>=t).astype(int))) for t in ths), key=lambda x: x[1])
print(f"OOF Accuracy = {best_acc:.4f} at threshold = {best_t:.3f}")

test_prob = test_pred_folds.mean(axis=0)
pred = (test_prob >= best_t).astype(int)

# ---- Optional: one-round pseudo-labeling (safe) ----
if PSEUDO_LABEL:
    high_pos = np.where(test_prob >= PL_POS_THR)[0]
    high_neg = np.where(test_prob <= PL_NEG_THR)[0]
    sel = np.concatenate([high_pos, high_neg])

    if sel.size > 0:
        X_aug = pd.concat([X_train, X_test.iloc[sel].copy()], axis=0, ignore_index=True)
        y_aug = pd.concat([
            y.reset_index(drop=True),
            pd.Series((test_prob[sel] >= PL_POS_THR).astype(int))
        ], axis=0).reset_index(drop=True)

        # retrain CatBoost with same features (do NOT recompute TE using pseudo labels)
        oof2 = np.zeros(len(X_train), dtype=float)
        test_pred_folds2 = np.zeros((N_SPLITS, len(X_test)), dtype=float)
        fold_acc2 = []

        for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
            tr_idx_aug = np.concatenate([tr, n_train + np.arange(len(sel))])  # add pseudo-labeled rows to training side
            X_tr2, y_tr2 = X_aug.iloc[tr_idx_aug], y_aug.iloc[tr_idx_aug]
            X_va2, y_va2 = X_train.iloc[va], y.iloc[va]

            est2 = CatBoostClassifier(
                depth=6, learning_rate=0.05, iterations=3000, l2_leaf_reg=3.0,
                loss_function="Logloss", eval_metric="Accuracy",
                random_seed=RANDOM_STATE+123+fold, od_type="Iter", od_wait=EARLY_STOPPING_ROUNDS, verbose=False
            )
            tr_pool2 = Pool(X_tr2, y_tr2, cat_features=cat_idx)
            va_pool2 = Pool(X_va2, y_va2, cat_features=cat_idx)
            te_pool2 = Pool(X_test, cat_features=cat_idx)

            est2.fit(tr_pool2, eval_set=va_pool2, use_best_model=True)
            va_p2 = est2.predict_proba(va_pool2)[:,1]
            te_p2 = est2.predict_proba(te_pool2)[:,1]

            oof2[va] = va_p2
            test_pred_folds2[fold-1,:] = te_p2
            fold_acc2.append(accuracy_score(y_va2, (va_p2>=0.5).astype(int)))

        print("PL CatBoost CV Acc per fold:", [f"{a:.4f}" for a in fold_acc2], "| Mean:", f"{np.mean(fold_acc2):.4f}")
        ths2 = np.linspace(0.3, 0.7, 401)
        best_t2, best_acc2 = max(((t, accuracy_score(y, (oof2>=t).astype(int))) for t in ths2), key=lambda x: x[1])
        print(f"[PL] OOF Accuracy = {best_acc2:.4f} at threshold = {best_t2:.3f}")
        test_prob = test_pred_folds2.mean(axis=0)
        pred = (test_prob >= best_t2).astype(int)
    else:
        print("No high-confidence test samples for pseudo-labeling; skipping.")

# ---- Submission ----
sub = pd.DataFrame({"PassengerId": test["PassengerId"].values, "Survived": pred})
sub.to_csv("submission.csv", index=False)
print("Saved: submission.csv", sub.shape)