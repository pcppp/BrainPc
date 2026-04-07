import json
from pathlib import Path

import importlib.util


def _load_metadata_utils():
    module_path = Path(__file__).resolve().parent / 'metadata_v5_utils.py'
    spec = importlib.util.spec_from_file_location('metadata_v5_utils', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metadata_utils = _load_metadata_utils()
DATASET_METADATA = _metadata_utils.DATASET_METADATA
encode_binary_label = _metadata_utils.encode_binary_label
encode_sex = _metadata_utils.encode_sex
get_subject_row = _metadata_utils.get_subject_row

ROOT = Path(__file__).resolve().parent


def build_meta(dataset_name: str):
    pattern = ROOT.parent / 'dataset' / dataset_name / '*' / '*_schaefer100_features_timeseries.mat'
    graph_paths = sorted(pattern.parent.glob(pattern.name))
    if not graph_paths:
        graph_paths = sorted((ROOT.parent / 'dataset' / dataset_name).glob('*/*_schaefer100_features_timeseries.mat'))
    if not graph_paths:
        raise FileNotFoundError(f'No schaefer100 feature files found for {dataset_name}')

    idx2sex = []
    idx2age = []
    idx2site = []
    idx2label = []
    missing = []

    for graph_path in graph_paths:
        subject_name = graph_path.name.replace('_schaefer100_features_timeseries.mat', '')
        row = get_subject_row(dataset_name, subject_name)
        if row is None:
            missing.append(subject_name)
            continue
        idx2sex.append(encode_sex(row['Sex']))
        idx2age.append(float(row['Age']))
        idx2site.append(0)
        idx2label.append(encode_binary_label(row['Group']))

    if missing:
        raise KeyError(f'Missing metadata rows for {dataset_name}: {missing[:10]}')

    meta = {
        'idx2sex': idx2sex,
        'idx2age': idx2age,
        'idx2site': idx2site,
        'idx2label': idx2label,
    }

    out_dir = ROOT / DATASET_METADATA[dataset_name]['meta_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'meta.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)

    labels = {'control': sum(1 for x in idx2label if x == 0), 'non_control': sum(1 for x in idx2label if x == 1)}
    print(f'[{dataset_name}] wrote {out_path} entries={len(idx2label)} labels={labels}')


if __name__ == '__main__':
    for dataset_name in DATASET_METADATA:
        build_meta(dataset_name)
