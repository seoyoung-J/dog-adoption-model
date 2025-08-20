"""
JSON -> (X.npy, y.npy, files_used.npy 생성)
- feature_labels == "Mouth Open" → 1, else → 0
- 존재하고 열리는 이미지만 사용 → RGB, 224x224, float32(0~1) 정규화
"""

import os, json, numpy as np
from PIL import Image

JSON_PATH  = "/content/project-4-ver3.json"   
IMAGE_DIR  = "/content/label-studio_img_1000"    # 이미지 폴더 경로
OUTPUT_X   = "/content/X_mouth.npy"              # 이미지 배열 저장 경로
OUTPUT_Y   = "/content/y_mouth.npy"              # 라벨 배열 저장 경로
OUTPUT_USED= "/content/files_used.npy"           # 사용된 파일명 순서 저장
IMG_SIZE   = (224, 224)

# 앞의 UUID 접두사 제거
def clean_name(s):
    base = os.path.basename(s)
    parts = base.split("-")

    return "-".join(parts[1:]) if len(parts) > 1 else base

# JSON 로드 & 파일명:라벨 매핑
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

fname_to_label = {}
for item in data:
    if "image" not in item:
        continue
    fn = clean_name(item["image"])
    lab = 1 if item.get("feature_labels") == "Mouth Open" else 0
    fname_to_label[fn] = lab

# 이미지 로드 및 전처리 
X_list, y_list, used_files = [], [], []
not_found, open_error = 0, 0

for item in data:
    if "image" not in item:
        continue
    fn = clean_name(item["image"])
    if fn not in fname_to_label:
        continue

    full_path = os.path.join(IMAGE_DIR, fn)
    if not os.path.exists(full_path):
        not_found += 1
        continue

    try:
        img = Image.open(full_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
    except Exception:
        open_error += 1
        continue

    X_list.append(arr)
    y_list.append(fname_to_label[fn])
    used_files.append(fn)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)
np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)
np.save(OUTPUT_USED, np.array(used_files, dtype=object))
