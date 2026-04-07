import csv
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

import importlib.util


def _load_metadata_utils():
    module_path = Path(__file__).resolve().parent / 'metadata_v5_utils.py'
    spec = importlib.util.spec_from_file_location('metadata_v5_utils', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metadata_utils = _load_metadata_utils()
DATASET_METADATA = _metadata_utils.DATASET_METADATA
get_dataset_meta_dir = _metadata_utils.get_dataset_meta_dir

ROOT = Path(__file__).resolve().parent
SPLIT_SECTIONS = ['train', 'ood_val', 'ood_test', 'id_val', 'id_test']


def write_index_file(path: Path, folds):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(folds)


def build_splits(dataset_name: str, random_state: int = 42):
    meta_dir = ROOT / get_dataset_meta_dir(dataset_name)
    meta_path = meta_dir / 'meta.json'
    with meta_path.open(encoding='utf-8') as f:
        meta = json.load(f)
    labels = meta['idx2label']
    indices = list(range(len(labels)))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    split_map = {name: [] for name in SPLIT_SECTIONS}
    for fold_id, (train_val_idx, ood_test_idx) in enumerate(skf.split(indices, labels)):
        train_val_labels = [labels[i] for i in train_val_idx]
        train_idx, ood_val_idx = train_test_split(
            train_val_idx,
            test_size=1 / 9,
            random_state=random_state + fold_id,
            stratify=train_val_labels,
        )
        train_labels = [labels[i] for i in train_idx]
        train_idx, id_holdout_idx = train_test_split(
            train_idx,
            test_size=0.2,
            random_state=random_state + 100 + fold_id,
            stratify=train_labels,
        )
        id_holdout_labels = [labels[i] for i in id_holdout_idx]
        id_val_idx, id_test_idx = train_test_split(
            id_holdout_idx,
            test_size=0.5,
            random_state=random_state + 200 + fold_id,
            stratify=id_holdout_labels,
        )

        split_map['train'].append(sorted(train_idx.tolist()))
        split_map['ood_val'].append(sorted(ood_val_idx.tolist()))
        split_map['ood_test'].append(sorted(ood_test_idx.tolist()))
        split_map['id_val'].append(sorted(id_val_idx.tolist()))
        split_map['id_test'].append(sorted(id_test_idx.tolist()))

    for section, folds in split_map.items():
        write_index_file(meta_dir / f'{section}.index', folds)
    print(f'[{dataset_name}] wrote split files under {meta_dir}')


if __name__ == '__main__':
    for dataset_name in DATASET_METADATA:
        build_splits(dataset_name)
