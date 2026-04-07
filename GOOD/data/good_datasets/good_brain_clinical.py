import csv
import json
import os
import os.path as osp

import torch
from dgl.data.utils import load_graphs
from munch import Munch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from GOOD import register


DATASET_CONFIG = {
    'GOODNEUROCON': {
        'display_name': 'Neurocon',
        'meta_name': 'neurocon_ood_schaefer100',
        'meta_json': './GOOD/data/good_datasets/neurocon_ood_schaefer100/meta.json',
        'bin_candidates': [
            './GOOD/data/bin_time_dataset/neurocon.bin',
            './GOOD/data/bin_dataset/neurocon.bin',
        ],
    },
    'GOODPPMI': {
        'display_name': 'PPMI',
        'meta_name': 'ppmi_metadata_ood_schaefer100',
        'meta_json': './GOOD/data/good_datasets/ppmi_metadata_ood_schaefer100/meta.json',
        'bin_candidates': [
            './GOOD/data/bin_time_dataset/ppmi.bin',
            './GOOD/data/bin_dataset/ppmi.bin',
        ],
    },
    'GOODTAOWU': {
        'display_name': 'TaoWu',
        'meta_name': 'taowu_ood_schaefer100',
        'meta_json': './GOOD/data/good_datasets/taowu_ood_schaefer100/meta.json',
        'bin_candidates': [
            './GOOD/data/bin_time_dataset/taowu.bin',
            './GOOD/data/bin_dataset/taowu.bin',
        ],
    },
}


class GOODBrainClinical(InMemoryDataset):
    dataset_key = None

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, data_list: list = None):
        self.name = self.__class__.__name__
        self.mol_name = DATASET_CONFIG[self.dataset_key]['display_name']
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Binary classification'
        self.generate = generate

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_dir(self):
        return osp.join(self.root)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'ood_val.pt', 'ood_test.pt', 'id_val.pt', 'id_test.pt']

    def process(self):
        print('#IN#Using default OOD splits')

    @classmethod
    def load(cls, dataset_root: str, domain: str = 'site', shift: str = 'no_shift', generate: bool = False, fold: int = 0):
        cfg = DATASET_CONFIG[cls.dataset_key]
        meta_info = Munch()
        meta_info.dataset_type = 'brain'
        meta_info.model_level = 'graph'
        meta_info.num_node_features = None
        meta_info.name = cfg['meta_name']
        meta_info.edge_ratio = 0.2
        meta_info.node_feat_transform = 'precomputed'

        with open(cfg['meta_json'], 'r') as f:
            meta_json = json.load(f)

        dataset_path = next((path for path in cfg['bin_candidates'] if osp.exists(path)), None)
        if dataset_path is None:
            raise FileNotFoundError(
                f"{cfg['display_name']} bin file not found. Expected one of: " + ', '.join(cfg['bin_candidates'])
            )
        print(f"#IN#Loading {cfg['display_name']} graphs from {dataset_path}")
        G_dataset, Labels = load_graphs(dataset_path)

        for i in tqdm(range(len(G_dataset))):
            if 'feat' not in G_dataset[i].edata:
                G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()
            if 'feat' not in G_dataset[i].ndata:
                if 'FC_features' in G_dataset[i].ndata:
                    G_dataset[i].ndata['feat'] = G_dataset[i].ndata['FC_features'].clone()
                else:
                    G_dataset[i].ndata['feat'] = G_dataset[i].ndata['N_features'].clone()

        all_idx = get_all_split_idx(meta_info.name)
        train_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx], meta_json[f'idx2{domain}'][idx]) for idx in all_idx['train'][fold]]
        id_val_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx], meta_json[f'idx2{domain}'][idx]) for idx in all_idx['id_val'][fold]]
        id_test_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx], meta_json[f'idx2{domain}'][idx]) for idx in all_idx['id_test'][fold]]
        val_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx], meta_json[f'idx2{domain}'][idx]) for idx in all_idx['ood_val'][fold]]
        test_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx], meta_json[f'idx2{domain}'][idx]) for idx in all_idx['ood_test'][fold]]

        train_dataset = cls(root=dataset_root, domain=domain, shift=shift, subset='train', generate=generate, data_list=train_data)
        id_val_dataset = cls(root=dataset_root, domain=domain, shift=shift, subset='id_val', generate=generate, data_list=id_val_data) if shift != 'no_shift' else None
        id_test_dataset = cls(root=dataset_root, domain=domain, shift=shift, subset='id_test', generate=generate, data_list=id_test_data) if shift != 'no_shift' else None
        val_dataset = cls(root=dataset_root, domain=domain, shift=shift, subset='val', generate=generate, data_list=val_data)
        test_dataset = cls(root=dataset_root, domain=domain, shift=shift, subset='test', generate=generate, data_list=test_data)

        meta_info.num_node_features = int(G_dataset[0].ndata['feat'].shape[-1])
        meta_info.dim_node = meta_info.num_node_features
        meta_info.dim_edge = 0
        meta_info.num_envs = torch.unique(torch.tensor(meta_json['idx2site']).long()).shape[0]
        meta_info.num_classes = 2

        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {
            'train': train_dataset,
            'id_val': id_val_dataset,
            'id_test': id_test_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'task': train_dataset.task,
            'metric': train_dataset.metric,
        }, meta_info


def get_all_split_idx(name):
    root_idx_dir = './GOOD/data/good_datasets/{}/'.format(name)
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}
    if not os.path.exists(root_idx_dir + 'train.index'):
        print('[!] no split at {}'.format(root_idx_dir))
        raise NotImplementedError
    for section in ['train', 'ood_val', 'ood_test', 'id_val', 'id_test']:
        with open(root_idx_dir + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx


def dgl_to_pyg(graph, y, domain):
    x = graph.ndata['feat']
    edge_index = torch.stack(graph.edges()).contiguous()
    edge_weight = graph.edata['feat'].float()
    yy = torch.zeros(1, 2)
    yy[0][int(y)] = 1
    data = Data(x=x.float(), edge_index=edge_index, edge_weight=edge_weight, y=yy, domain=domain)
    data.env_id = domain
    return data


@register.dataset_register
class GOODNEUROCON(GOODBrainClinical):
    dataset_key = 'GOODNEUROCON'


@register.dataset_register
class GOODPPMI(GOODBrainClinical):
    dataset_key = 'GOODPPMI'


@register.dataset_register
class GOODTAOWU(GOODBrainClinical):
    dataset_key = 'GOODTAOWU'
