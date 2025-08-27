import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (precision_recall_curve, roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, classification_report)
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/dog_data_final.csv")

target_column = "adoption_status"
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rows = []
order_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

rf  = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_rf  = cross_val_predict(rf,  X_train, y_train, cv=cv5, method="predict_proba", n_jobs=-1)[:, 1]
oof_xgb = cross_val_predict(xgb, X_train, y_train, cv=cv5, method="predict_proba", n_jobs=-1)[:, 1]

# 가중치 α + 임계값 최적화 
alphas = np.linspace(0.0, 1.0, 101)
recall_floor = 0.75
eps = 0.005
stats = []

for a in alphas:
    oof_blend = a * oof_rf + (1 - a) * oof_xgb
    p, r, t = precision_recall_curve(y_train, oof_blend)
    if t.size == 0:
        continue
    f1s = 2 * p[1:] * r[1:] / (p[1:] + r[1:] + 1e-12)

    if (r[1:] >= recall_floor).any():
        mask = r[1:] >= recall_floor
        k = int(np.argmax(f1s[mask]))
        thr = float(t[mask][k])
        f1  = float(f1s[mask][k])
    else:
        k = int(np.argmax(f1s))
        thr = float(t[k])
        f1  = float(f1s[k])

    stats.append((a, thr, f1, roc_auc_score(y_train, oof_blend)))

alpha, thr, f1_oof, auc_oof = max(stats, key=lambda s: (s[2], s[3]))
print(f"[Soft Voting] Picked alpha={alpha:.2f}, thr={thr:.4f} (OOF F1={f1_oof:.4f}, AUC={auc_oof:.4f})")

# Train 재학습 및 평가 
rf_f  = clone(rf).fit(X_train, y_train)
xgb_f = clone(xgb).fit(X_train, y_train)

prob_rf  = rf_f.predict_proba(X_test)[:, 1]
prob_xgb = xgb_f.predict_proba(X_test)[:, 1]
prob_blend = alpha * prob_rf + (1 - alpha) * prob_xgb

y_pred_sv = (prob_blend >= thr).astype(int)
print("\n=== Soft Voting (RF+XGB) : TEST ===")
print(classification_report(y_test, y_pred_sv, digits=4))
print(f"Accuracy : {accuracy_score(y_test, y_pred_sv):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_sv):.4f}")
print(f"Recall : {recall_score(y_test, y_pred_sv):.4f}  (target floor={recall_floor})")
print(f"F1 : {f1_score(y_test, y_pred_sv):.4f}")
print(f"ROC-AUC : {roc_auc_score(y_test, prob_blend):.4f}")

rows.append({
    "Model": f"SoftVote RF+XGB (α={alpha:.2f}, thr={thr:.2f})",
    "Accuracy": accuracy_score(y_test, y_pred_sv),
    "Precision": precision_score(y_test, y_pred_sv),
    "Recall": recall_score(y_test, y_pred_sv),
    "F1": f1_score(y_test, y_pred_sv),
    "ROC-AUC": roc_auc_score(y_test, prob_blend)
})
df_results = pd.DataFrame(rows)[order_cols].sort_values(by=["F1", "ROC-AUC"], ascending=False)
print("\n=== Model Comparison (with Soft Voting) ===")
print(df_results.round(4))
