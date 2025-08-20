import os, sys, json, platform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy.random import default_rng
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import AdamW


#=============== 공용 함수 정의 (split/oversample/threshold) ==========

# 70/15/15 고정 분할 인덱스 생성
def make_split_indices(y, seed=42):
    y = np.asarray(y).ravel().astype(int)

    test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    (train_val_idx, test_idx), = test_splitter.split(np.zeros_like(y), y)

    val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=seed)
    (train_rel_idx, val_rel_idx), = val_splitter.split(np.zeros_like(train_val_idx), y[train_val_idx])

    train_idx = train_val_idx[train_rel_idx]
    val_idx   = train_val_idx[val_rel_idx]

    return train_idx, val_idx, test_idx

# Train 전용 1:1 오버샘플링 함수 
def make_balanced_train_indices(y_train, seed=42):
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train).ravel().astype(int)

    neg_idx = np.where(y_train == 0)[0]
    pos_idx = np.where(y_train == 1)[0]

    if len(neg_idx) > len(pos_idx):
        need = len(neg_idx) - len(pos_idx)
        rep  = rng.choice(pos_idx, size=need, replace=True)
        bal_idx = np.concatenate([neg_idx, pos_idx, rep])
    elif len(pos_idx) > len(neg_idx):
        need = len(pos_idx) - len(neg_idx)
        rep  = rng.choice(neg_idx, size=need, replace=True)
        bal_idx = np.concatenate([pos_idx, neg_idx, rep])
    else:
        bal_idx = np.arange(len(y_train))
    rng.shuffle(bal_idx)

    return bal_idx

# 임계값 선택 함수 
def pick_threshold(y_true, prob, policy="f1_max", target_p=0.65, default=0.5):
    y_true = np.asarray(y_true).ravel().astype(int)
    prob   = np.asarray(prob).ravel()

    if len(np.unique(y_true)) < 2:
        return float(default), f"default({default}) - single class in y_true"
    if policy == "fixed_0.5":
        return 0.5, "fixed_0.5"
    p, r, t = precision_recall_curve(y_true, prob)  
    if t.size == 0: 
        return float(default), f"default({default}) - no thresholds"
    p_, r_, t_ = p[1:], r[1:], t 
    if policy == "precision_at":
        ok = np.where(p_ >= float(target_p))[0]
        if len(ok):
            k = ok[np.argmax(r_[ok])]
            return float(t_[k]), f"P≥{target_p} maxR (P={p_[k]:.3f}, R={r_[k]:.3f})"
        policy = "f1_max" 
    f1 = 2 * p_ * r_ / (p_ + r_ + 1e-12)  
    k  = int(np.argmax(f1))

    return float(t_[k]), f"F1-max (P={p_[k]:.3f}, R={r_[k]:.3f})"

def get_model_result(model_name, y_val, val_prob, y_test, test_prob, thr, settings):
    row = (
        f"| {model_name} | "
        f"{average_precision_score(y_val, val_prob):.3f} / {average_precision_score(y_test, test_prob):.3f} | "
        f"{roc_auc_score(y_val, val_prob):.3f} / {roc_auc_score(y_test, test_prob):.3f} | "
        f"{f1_score(y_test, (test_prob >= thr).astype(int), zero_division=0):.3f} | "
        f"{precision_score(y_test, (test_prob >= thr).astype(int), zero_division=0):.3f} / "
        f"{recall_score(y_test, (test_prob >= thr).astype(int), zero_division=0):.3f} | "
        f"{thr:.3f} | {settings} |"
    )
    return row

def dist(name, yy):
    u,c = np.unique(yy, return_counts=True)
    print(f"{name} dist:", dict(zip(u,c)))

# 분할 인덱스 생성 & 저장 
X = np.load("/content/X_mouth.npy")
y = np.load("/content/y_mouth.npy").astype(np.int32)
train_idx, val_idx, test_idx = make_split_indices(y, seed=42)
np.save("/content/train_idx.npy", train_idx)
np.save("/content/val_idx.npy", val_idx)
np.save("/content/test_idx.npy", test_idx)

print("SPLIT SIZES:",len(train_idx), len(val_idx), len(test_idx))
print("DIST TRAIN:", dict(zip(*np.unique(y[train_idx], return_counts=True))))
print("DIST VAL  :", dict(zip(*np.unique(y[val_idx], return_counts=True))))
print("DIST TEST :", dict(zip(*np.unique(y[test_idx], return_counts=True))))


# ======== 모델 학습 =======
opt_head = AdamW(learning_rate=3e-4, weight_decay=1e-4, clipnorm=1.0)
opt_ft   = AdamW(learning_rate=1e-5, weight_decay=1e-4, clipnorm=1.0)

# seed 설정
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 데이터 로드
X = np.load("/content/X_mouth.npy")
y = np.load("/content/y_mouth.npy").astype(np.int32)
assert X.ndim == 4 and X.shape[1:] == (224,224,3), f"Unexpected X shape: {X.shape}"
assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"

# train/val/test(70/15/15) 분할
train_idx = np.load("/content/train_idx.npy")
val_idx   = np.load("/content/val_idx.npy")
test_idx  = np.load("/content/test_idx.npy")

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# Oversampling (train only)
bal_idx = make_balanced_train_indices(y_train, seed=SEED)
X_train_bal, y_train_bal = X_train[bal_idx], y_train[bal_idx]
dist("train_bal", y_train_bal)

# 데이터 증강
augment_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
]
augment = tf.keras.Sequential(augment_layers, name="augment")

# ResNet50V2
def build_resnet50v2_model():
    inputs = layers.Input((224, 224, 3))
    x = augment(inputs)
    x = layers.Rescaling(255.0)(x)  # (0~1) -> (0~255)
    x = layers.Lambda(preprocess_input, name="imagenet_preprocess")(x)  # [-1,1]

    base = ResNet50V2(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.40)(x) # 0.30 -> 0.40
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model

model = build_resnet50v2_model()

# Phase 1 (BCE, AdamW)
model.compile(
    optimizer=opt_head,
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
    metrics=["accuracy", AUC(name="auc"), AUC(name="ap", curve="PR")],
)
es1  = EarlyStopping(monitor="val_ap", mode="max", patience=3, restore_best_weights=True)
rlr1 = ReduceLROnPlateau(monitor="val_ap", mode="max", factor=0.5, patience=1, min_lr=1e-6)

print("=== Phase 1: Quick head adapt (BCE) ===")
_ = model.fit(
    X_train_bal, y_train_bal,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    callbacks=[es1, rlr1],
    verbose=1
)

# Unfreeze (32 layers)
base = None
for lyr in model.layers:
    if isinstance(lyr, tf.keras.Model) and lyr.name.startswith("resnet50v2"):
        base = lyr; break

if base is not None:
    trainable_count = 0
    for lyr in base.layers[-32:]:
        if isinstance(lyr, layers.BatchNormalization):
            lyr.trainable = False
        else:
            lyr.trainable = True
            trainable_count += 1
    print(f"Unfreezing top layers (excluding BN): {trainable_count}")

else:
    raise RuntimeError("ResNet50V2 base 레이어를 찾을 수 없습니다.")

# Phase 2 (BCE, AdamW, patience ↑)
model.compile(
    optimizer=opt_ft,
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
    metrics=["accuracy", AUC(name="auc"), AUC(name="ap", curve="PR")],
)
es2  = EarlyStopping(monitor="val_ap", mode="max", patience=8, restore_best_weights=True)  # 6 → 8
rlr2 = ReduceLROnPlateau(monitor="val_ap", mode="max", factor=0.5, patience=3, min_lr=1e-6)

print("=== Phase 2: Fine-tuning top layers (BCE) ===")
_ = model.fit(
    X_train_bal, y_train_bal,
    validation_data=(X_val, y_val),
    epochs=20, batch_size=16,
    callbacks=[es2, rlr2],
    verbose=1
)

# 평가/임계값
val_prob  = model.predict(X_val,  verbose=0).ravel()
test_prob = model.predict(X_test, verbose=0).ravel()
thr, info = pick_threshold(y_val, val_prob, policy="f1_max")

print("[VAL] ROC-AUC / PR-AUC:", roc_auc_score(y_val, val_prob),  average_precision_score(y_val, val_prob))
print("[TEST] ROC-AUC / PR-AUC:", roc_auc_score(y_test, test_prob), average_precision_score(y_test, test_prob))
print(f"[VAL] threshold={thr:.3f} ({info})")

yhat = (test_prob >= thr).astype(int)
print("\n[TEST metrics @thr-selected]")
print("accuracy:", (yhat == y_test).mean())
print(classification_report(y_test, yhat, digits=3))

row = get_model_result(
    model_name="ResNet50V2 (transfer, AdamW)",
    y_val=y_val, val_prob=val_prob,
    y_test=y_test, test_prob=test_prob,
    thr=thr,
    settings=(
        "oversample(1:1), BCE, AdamW(wd=1e-4, clipnorm=1.0), "
        "GAP, Dropout=0.4, aug=flip+rot0.05+zoom0.10+contrast0.10, "
        "unfreeze_last=32, thr_policy="
        f"{info}")
)
print("\n[model_result]\n" + row)


# ======= 학습 결과 저장 ========
BASE_DIR = "/content/artifacts-v3"
os.makedirs(BASE_DIR, exist_ok=True)

assert 'model' in globals() and 'thr' in globals(), "First, run the learning cell to define model, thr, etc."

KERAS_PATH = os.path.join(BASE_DIR, "model_best.keras")      # 전체 모델
WEIGHTS_H5 = os.path.join(BASE_DIR, "model.weights.h5")      # 가중치
ARCH_JSON  = os.path.join(BASE_DIR, "model_arch.json")       # 아키텍처

model.save(KERAS_PATH)
model.save_weights(WEIGHTS_H5)
print("[OK] Saved full model (.keras):", KERAS_PATH)
print("[OK] Saved weights (.weights.h5):", WEIGHTS_H5)
try:
    with open(ARCH_JSON, "w") as f:
        f.write(model.to_json())
    print("[OK] Saved architecture (.json):", ARCH_JSON)
except Exception as e:
    print("[WARN] model.to_json() 실패:", e)
THR_PATH = os.path.join(BASE_DIR, "best_threshold.txt")
with open(THR_PATH, "w") as f:
    f.write(f"{thr:.6f}")
np.save(os.path.join(BASE_DIR, "val_prob.npy"),  val_prob)
np.save(os.path.join(BASE_DIR, "test_prob.npy"), test_prob)
np.save(os.path.join(BASE_DIR, "y_val.npy"),     y_val)
np.save(os.path.join(BASE_DIR, "y_test.npy"),    y_test)
print("Saved threshold & npy arrays.")

# 메트릭 요약 JSON / 리포트 / 혼동행렬 
yhat = (test_prob >= thr).astype(int)
metrics = {
    "threshold": float(thr),
    "val_roc_auc":  float(roc_auc_score(y_val, val_prob)),
    "val_pr_auc":   float(average_precision_score(y_val, val_prob)),
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_pr_auc":  float(average_precision_score(y_test, test_prob)),
    "test_precision_at_thr": float(precision_score(y_test, yhat, zero_division=0)),
    "test_recall_at_thr":    float(recall_score(y_test, yhat, zero_division=0)),
    "test_f1_at_thr":        float(f1_score(y_test, yhat, zero_division=0)),
}
with open(os.path.join(BASE_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
rep = classification_report(y_test, yhat, digits=4)
with open(os.path.join(BASE_DIR, "test_classification_report.txt"), "w") as f:
    f.write(rep)
cm = confusion_matrix(y_test, yhat, labels=[0,1])
np.savetxt(os.path.join(BASE_DIR, "test_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
print("Saved metrics json, report txt, confusion-matrix csv.")

# PR/ROC 커브 PNG 
ps, rs, _ = precision_recall_curve(y_test, test_prob)
fpr, tpr, _ = roc_curve(y_test, test_prob)
plt.figure(); plt.plot(rs, ps)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve (TEST)")
plt.savefig(os.path.join(BASE_DIR, "pr_curve_test.png"), dpi=180); plt.close()
plt.figure(); plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC curve (TEST)")
plt.savefig(os.path.join(BASE_DIR, "roc_curve_test.png"), dpi=180); plt.close()
print("Saved PR/ROC curves.")

print("\n== DONE ==")
print("Artifacts saved to:", BASE_DIR)