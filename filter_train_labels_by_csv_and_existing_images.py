import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "my_json"
TRAIN_JSON_PATH = JSON_DIR / "train_labels.json"
TEST_JSON_PATH = JSON_DIR / "test_labels.json"
TRAIN_CSV_PATH = BASE_DIR / "my_csv_train.csv"
TEST_CSV_PATH = BASE_DIR / "my_csv_test.csv"
IMAGE_ROOT = Path(r"D:\WorkFiles\ThyroidAgent\Data\Malignant_ultrasound_images_cropped_deleted")
TRAIN_OUTPUT_JSON_PATH = JSON_DIR / "train_labels_filtered_by_csv_and_existing_images.json"
TEST_OUTPUT_JSON_PATH = JSON_DIR / "test_labels_filtered_by_csv_and_existing_images.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_patient_ids(csv_path: Path) -> set[str]:
    patient_ids = set()
    encodings = ["utf-8-sig", "gbk", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    for value in row[:2]:
                        value = value.strip()
                        if value:
                            patient_ids.add(value.replace("\\", "/"))
            return patient_ids
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error if last_error else ValueError(f"无法读取 {csv_path}")


def build_existing_relative_paths(image_root: Path) -> set[str]:
    relative_paths = set()
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_paths.add(path.relative_to(image_root).as_posix())
    return relative_paths


def extract_patient_id(filename: str) -> str:
    parts = filename.replace("\\", "/").split("/")
    if len(parts) < 3:
        raise ValueError(f"无法从 filename 提取患者ID: {filename}")
    return f"{parts[0]}/{parts[1]}"


def filter_labels(json_path: Path, patient_ids: set[str], existing_paths: set[str]) -> tuple[list[dict], int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path.name} 顶层必须是 list")

    filtered = []
    for item in data:
        if not isinstance(item, dict) or "filename" not in item:
            continue

        filename = str(item["filename"]).replace("\\", "/")
        patient_id = extract_patient_id(filename)

        if patient_id not in patient_ids:
            continue
        if filename not in existing_paths:
            continue

        filtered.append(item)

    return filtered, len(data)


def process_dataset(json_path: Path, csv_path: Path, output_path: Path, existing_paths: set[str]) -> None:
    patient_ids = load_patient_ids(csv_path)
    filtered, total_count = filter_labels(json_path, patient_ids, existing_paths)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"{json_path.name}: {total_count} -> {len(filtered)}")
    print(f"{csv_path.name} 患者数: {len(patient_ids)}")
    print(f"输出文件: {output_path}")


def main() -> None:
    existing_paths = build_existing_relative_paths(IMAGE_ROOT)
    print(f"目录中实际图像数: {len(existing_paths)}")

    process_dataset(
        TRAIN_JSON_PATH,
        TRAIN_CSV_PATH,
        TRAIN_OUTPUT_JSON_PATH,
        existing_paths,
    )
    process_dataset(
        TEST_JSON_PATH,
        TEST_CSV_PATH,
        TEST_OUTPUT_JSON_PATH,
        existing_paths,
    )


if __name__ == "__main__":
    main()
