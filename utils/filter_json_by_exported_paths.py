from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize_path(value: str) -> str:
    return str(value).replace('\\', '/').strip().strip('/')


def build_match_keys(value: str) -> set[str]:
    normalized = normalize_path(value)
    if not normalized:
        return set()

    keys = {normalized}
    parts = normalized.split('/')
    if len(parts) >= 3:
        keys.add('/'.join(parts[1:]))
    return keys


def load_exported_paths(patient_summary_csv: Path) -> set[str]:
    exported_paths = set()
    with open(patient_summary_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_paths = (row.get('exported_relative_paths') or '').strip()
            if not raw_paths:
                continue
            for path in raw_paths.split('|'):
                exported_paths.update(build_match_keys(path))
    return exported_paths


def filter_json_records(input_json: Path, exported_paths: set[str]) -> tuple[list[dict], int]:
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError('Input JSON top level must be a list')

    filtered = []
    removed = 0
    for item in data:
        if not isinstance(item, dict):
            removed += 1
            continue
        filename = item.get('filename', '')
        if build_match_keys(filename) & exported_paths:
            filtered.append(item)
        else:
            removed += 1
    return filtered, removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Filter a JSON label list by exported_relative_paths from patient_summary.csv'
    )
    parser.add_argument(
        '--patient_summary_csv',
        type=str,
        required=True,
        help='Path to patient_summary.csv',
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to input JSON, e.g. my_json/train_labels_sample.json',
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='',
        help='Optional output path; defaults to <input_stem>_exported_only.json',
    )
    args = parser.parse_args()

    patient_summary_csv = Path(args.patient_summary_csv)
    input_json = Path(args.input_json)
    output_json = Path(args.output_json) if args.output_json else input_json.with_name(f'{input_json.stem}_exported_only.json')

    if not patient_summary_csv.is_file():
        raise FileNotFoundError(f'patient_summary.csv not found: {patient_summary_csv}')
    if not input_json.is_file():
        raise FileNotFoundError(f'input_json not found: {input_json}')

    exported_paths = load_exported_paths(patient_summary_csv)
    filtered, removed = filter_json_records(input_json, exported_paths)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f'Loaded exported paths: {len(exported_paths)}')
    print(f'Input records: {len(filtered) + removed}')
    print(f'Kept records: {len(filtered)}')
    print(f'Removed records: {removed}')
    print(f'Output JSON: {output_json}')


if __name__ == '__main__':
    main()
