import json
import os
import numpy as np
from PIL import Image

json_path = "./project-3-at-2025-01-29-10-41-e546c10e.json"
image_directory = "./all_dogs_img"
output_npy_path = "./train_500_images.npy"


# 이미지 파일명 추출 (앞의 UUID 접두사 제거)
def load_image_filenames_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    image_paths = []
    for item in annotations:
        if "image" in item:
            original_filename = os.path.basename(item["image"])  
            cleaned_filename = "-".join(original_filename.split("-")[1:])  
            image_paths.append(cleaned_filename)

    print(f"Extracted {len(image_paths)} cleaned image filenames from JSON")
    if image_paths:
        print("Sample filenames:", image_paths[:5])

    return image_paths


# 이미지 로드 → RGB → 리사이즈(224x224) → 0~1정규화 
def preprocess_images(image_paths, image_directory, size=(224, 224)):
    processed_images = []
    valid_paths = []

    for filename in image_paths:
        full_path = os.path.join(image_directory, filename)

        try:
            img = Image.open(full_path).convert('RGB')
            img = img.resize(size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            processed_images.append(img_array)
            valid_paths.append(full_path)
        except FileNotFoundError: 
            print(f"Warning: File not found - {full_path}")
        except Exception as e:
            print(f"Error processing {full_path}: {e}")

    processed_images_array = np.array(processed_images)
    print(f"Processed {processed_images_array.shape[0]} images.")

    return processed_images_array, valid_paths


def main():
    image_paths = load_image_filenames_from_json(json_path)
    processed_images_array, valid_image_paths = preprocess_images(image_paths, image_directory)

    np.save(output_npy_path, processed_images_array)
    print(f"Saved preprocessed images to {output_npy_path}")

if __name__ == "__main__":
    main()
