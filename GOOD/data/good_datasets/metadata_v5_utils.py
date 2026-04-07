import csv
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META_DIR = ROOT / 'Metadata_v5'

DATASET_METADATA = {
    'neurocon': {
        'csv': 'Neurocon_metadata.csv',
        'meta_dir': 'neurocon_ood_schaefer100',
        'subject_mode': 'full',
    },
    'ppmi': {
        'csv': 'PPMI_metadata.csv',
        'meta_dir': 'ppmi_metadata_ood_schaefer100',
        'subject_mode': 'digits',
    },
    'taowu': {
        'csv': 'TaoWu_metadata.csv',
        'meta_dir': 'taowu_ood_schaefer100',
        'subject_mode': 'full',
    },
}

SEX_MAP = {
    'M': 1,
    'MALE': 1,
    'F': 2,
    'FEMALE': 2,
}


def has_metadata_v5(dataset_name: str) -> bool:
    return dataset_name in DATASET_METADATA


def get_dataset_meta_dir(dataset_name: str) -> str:
    return DATASET_METADATA[dataset_name]['meta_dir']


def extract_subject_key(subject_name: str, mode: str) -> str:
    subject_name = subject_name.replace('sub-', '')
    if mode == 'full':
        return subject_name
    if mode == 'digits':
        match = re.search(r'(\d+)$', subject_name)
        if not match:
            raise ValueError(f'Cannot extract numeric subject id from {subject_name}')
        return match.group(1)
    raise ValueError(f'Unsupported subject mode: {mode}')


@lru_cache(maxsize=None)
def load_metadata_rows(dataset_name: str):
    if dataset_name not in DATASET_METADATA:
        raise KeyError(f'Unsupported Metadata-V5 dataset: {dataset_name}')
    csv_path = META_DIR / DATASET_METADATA[dataset_name]['csv']
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    by_subject = {}
    for row in rows:
        subject = str(row['Subject']).strip()
        by_subject.setdefault(subject, row)
    return rows, by_subject


def get_subject_row(dataset_name: str, subject_name: str):
    if dataset_name not in DATASET_METADATA:
        return None
    _, by_subject = load_metadata_rows(dataset_name)
    mode = DATASET_METADATA[dataset_name]['subject_mode']
    subject_key = extract_subject_key(subject_name, mode)
    return by_subject.get(subject_key)


def encode_sex(value: str) -> int:
    key = str(value).strip().upper()
    if key not in SEX_MAP:
        raise ValueError(f'Unsupported sex value: {value}')
    return SEX_MAP[key]


def encode_binary_label(group: str) -> int:
    return 0 if str(group).strip().lower() == 'control' else 1
