import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report,
    precision_score, recall_score, make_scorer) 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv("/content/dog_data_final.csv")

target_column = "adoption_status"
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 스케일링(KNN의 경우) 및 모델 정의 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
    "LightGBM": LGBMClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorers = {
    "acc":  "accuracy",
    "prec": make_scorer(precision_score, pos_label=1),
    "rec":  make_scorer(recall_score,    pos_label=1),
    "f1":   make_scorer(f1_score,        pos_label=1),
    "auc":  "roc_auc",
}

# 교차검증 
models_cv = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost":      XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
    "LightGBM":     LGBMClassifier(random_state=42),
    "KNN":          Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
}

cv_rows = []
for name, mdl in models_cv.items():
    res = cross_validate(mdl, X_train, y_train, scoring=scorers, cv=cv, n_jobs=-1)
    cv_rows.append({
        "Model": name,
        "ACC(mean)":  res["test_acc"].mean(),
        "Prec(mean)": res["test_prec"].mean(),
        "Rec(mean)":  res["test_rec"].mean(),
        "F1(mean)":   res["test_f1"].mean(),
        "AUC(mean)":  res["test_auc"].mean(),
    })

df_cv = pd.DataFrame(cv_rows).sort_values(by=["F1(mean)", "AUC(mean)"], ascending=False)
print("\n=== 5-fold Stratified CV (train only) ===")
print(df_cv.round(4))

# 모델별 성능
rows = []
for name, model in models.items():
    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

order_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
df_results = pd.DataFrame(rows)[order_cols].sort_values(by=["F1", "ROC-AUC"], ascending=False)
print("\n=== Model Comparison ===")
print(df_results.round(4))
