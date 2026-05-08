"""
The GOOD-HIV dataset adapted from `MoleculeNet
<https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.
"""
import itertools
import os
import os.path as osp
import random
from copy import deepcopy
from dgl.data.utils import load_graphs
import gdown
import numpy as np
import torch
from munch import Munch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.datasets import MoleculeNet
from tqdm import tqdm
import csv
import json

class DomainGetter():
    r"""
    A class containing methods for data domain extraction.
    """

    def __init__(self):
        pass

    def get_scaffold(self, smile: str) -> str:
        """
        Args:
            smile (str): A smile string for a molecule.
        Returns:
            The scaffold string of the smile.
        """
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smile), includeChirality=False)
            return scaffold
        except ValueError as e:
            print('Get scaffold error.')
            raise e

    def get_nodesize(self, smile: str) -> int:
        """
        Args:
            smile (str): A smile string for a molecule.
        Returns:
            The number of node in the molecule.
        """
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            print('GetNumAtoms error, smiles:{}'.format(smile))
            return len(smile)
        number_atom = mol.GetNumAtoms()
        return number_atom


from GOOD import register


@register.dataset_register
class GOODABIDE(InMemoryDataset):
    r"""
    The GOOD-HIV dataset. Adapted from `MoleculeNet
    <https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, data_list: list = None):

        self.name = self.__class__.__name__
        self.mol_name = 'ABIDE'
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = ''

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        shift_mode = {'no_shift': 0, 'covariate': 3, 'concept': 8}
        mode = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        subset_pt = shift_mode[shift] + mode[subset]
        # self.data, self.slices = torch.load(self.processed_paths[subset_pt])
        # print(data_list[0].edge_index)
        # print(data_list[1].edge_index)
        self.data, self.slices = self.collate(data_list)
        # self.data = data_list

    @property
    def raw_dir(self):
        return osp.join(self.root)

    # def _download(self):
    #     if os.path.exists(osp.join(self.raw_dir, self.name)) or self.generate:
    #         return
    #     if not os.path.exists(self.raw_dir):
    #         os.makedirs(self.raw_dir)
    #     self.download()
    #
    # def download(self):
    #     path = gdown.download(self.url, output=osp.join(self.raw_dir, self.name + '.zip'), fuzzy=True)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'ood_val.pt', 'ood_test.pt', 'id_val.pt', 'id_test.pt',]


    def process(self):
        print('#IN#Using default OOD splits')


    @staticmethod
    def load(dataset_root: str, domain: str="site", shift: str = 'no_shift', generate: bool = False, fold: int = 0, protocol: str = "loso", **kwargs):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        meta_info = Munch()
        meta_info.dataset_type = 'brain'
        meta_info.model_level = 'graph'
        meta_info.num_node_features = None
        meta_info.name = 'abide_full_ood_schaefer100'
        meta_info.edge_ratio = -1  # soft edges, no hard threshold
        meta_info.node_feat_transform = 'precomputed'
        
        with open('./GOOD/data/good_datasets/abide_full_ood_schaefer100/meta.json', 'r') as f:
            meta_json = json.load(f)
        
       
        dataset_candidates = [
            './GOOD/data/bin_time_dataset/abide.bin',
            './GOOD/data/bin_dataset/abide.bin',
        ]
        dataset_path = next((path for path in dataset_candidates if osp.exists(path)), None)
        if dataset_path is None:
            raise FileNotFoundError(
                'ABIDE bin file not found. Expected one of: '
                + ', '.join(dataset_candidates)
            )
        print(f'#IN#Loading ABIDE graphs from {dataset_path}')
        G_dataset, Labels = load_graphs(dataset_path)


        error_case = []
        min_feat_dim = G_dataset[0].ndata['N_features'].shape[-1]
        for i in range(len(G_dataset)):
            if len(((G_dataset[i].ndata['N_features'] != 0).sum(dim=-1) == 0).nonzero()) > 0:
                error_case.append(i)
            if G_dataset[i].ndata['N_features'].shape[-1] < min_feat_dim:
                min_feat_dim = G_dataset[i].ndata['N_features'].shape[-1]
        print(error_case)
        # G_dataset = [n for i, n in enumerate(G_dataset) if i not in error_case]

        # 稀疏化
        # for i in tqdm(range(len(G_dataset))):
            
        #     # if edge_ratio:
        #     threshold_idx = int(len(G_dataset[i].edata['E_features']) * (1 - meta_info.edge_ratio))
        #     threshold = sorted(G_dataset[i].edata['E_features'].tolist())[threshold_idx]

        #     G_dataset[i].remove_edges(torch.squeeze((torch.abs(G_dataset[i].edata['E_features']) < float(threshold)).nonzero()))
        #     # G_dataset[i].edata['E_features'][G_dataset[i].edata['E_features'] < 0] = 0
        #     G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()

        #     if meta_info.node_feat_transform == 'pearson':
        #         G_dataset[i].ndata['feat'] = G_dataset[i].ndata['N_features'].clone()
        #         # G_dataset[i].ndata['feat'] = torch.from_numpy(np.corrcoef(G_dataset[i].ndata['N_features'].numpy())).clone()
        #     else:
        #         raise NotImplementedError
        
        # 优先使用预处理脚本写入的特征视图，缺失时再回退到旧字段。
        for i in tqdm(range(len(G_dataset))):
            if 'feat' not in G_dataset[i].edata:
                G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()

            if 'feat' not in G_dataset[i].ndata:
                if 'FC_features' in G_dataset[i].ndata:
                    G_dataset[i].ndata['feat'] = G_dataset[i].ndata['FC_features'].clone()
                else:
                    G_dataset[i].ndata['feat'] = G_dataset[i].ndata['N_features'].clone()

        # --- Build train/id_val/id_test/ood_val/ood_test indices ---
        # "loso":  S-fold leave-one-site-out (held-out site = OOD test, second
        #          site = OOD val, remaining S-2 sites split 80/10/10).
        # "10fold": legacy cached *.index files at the GOOD dataset dir.
        if str(protocol).lower() == 'loso':
            seed_for_split = 0  # deterministic across runs; only the within-site
                                # subject shuffle uses it
            all_idx, num_loso_folds, loso_sites = _build_loso_splits(meta_json, seed=seed_for_split)
            meta_info.protocol = 'loso'
            meta_info.num_folds = num_loso_folds
            meta_info.loso_sites = loso_sites
            print(f'#IN# LOSO protocol: {num_loso_folds} folds over sites {loso_sites}; '
                  f'fold {fold} -> ood_test_site={loso_sites[fold]}, '
                  f'ood_val_site={loso_sites[(fold + 1) % num_loso_folds]}')
        else:
            all_idx = get_all_split_idx(meta_info.name)
            meta_info.protocol = '10fold'
            meta_info.num_folds = 10
        # Fit PCA only on the training subjects of THIS fold.  Fitting on the
        # full dataset (legacy behaviour) leaks the OOD test/val site's
        # connectivity covariance into the principal-component basis, even
        # though the PCA itself is unsupervised.  Under LOSO that leak biases
        # OOD scores upward by 1-3 points.
        train_idx_for_pca = list(all_idx['train'][fold])
        fit_pca_on_train([G_dataset[i] for i in train_idx_for_pca])
        train_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['train'][fold]]
        id_val_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['id_val'][fold]]
        id_test_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['id_test'][fold]]
        val_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['ood_val'][fold]]
        test_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['ood_test'][fold]]

        train_dataset = GOODABIDE(root=dataset_root,
                                domain=domain, shift=shift, subset='train', generate=generate, data_list=train_data)
        id_val_dataset = GOODABIDE(root=dataset_root,
                                 domain=domain, shift=shift, subset='id_val',
                                 generate=generate, data_list=id_val_data) if shift != 'no_shift' else None
        id_test_dataset = GOODABIDE(root=dataset_root,
                                  domain=domain, shift=shift, subset='id_test',
                                  generate=generate, data_list=id_test_data) if shift != 'no_shift' else None
        val_dataset = GOODABIDE(root=dataset_root,
                              domain=domain, shift=shift, subset='val', generate=generate, data_list=val_data)
        test_dataset = GOODABIDE(root=dataset_root,
                               domain=domain, shift=shift, subset='test', generate=generate, data_list=test_data)

        # Use actual x dim from converted PyG data (may differ from raw due to PCA)
        meta_info.num_node_features = int(train_data[0].x.shape[-1])
        meta_info.dim_node = meta_info.num_node_features
        meta_info.dim_edge = 0 #train_dataset.num_edge_features

        # meta_info.num_envs = torch.unique(train_dataset.data.env_id).shape[0]
        meta_info.num_envs = torch.unique(torch.tensor(meta_json["idx2site"]).long()).shape[0]

        # Define networks' output shape.
        if train_dataset.task == 'Binary classification':
            meta_info.num_classes = 2  # train_dataset.data.y.shape[1]
        elif train_dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif train_dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(train_dataset.data.y).shape[0]

        # --- clear buffer dataset._data_list ---
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset, 'task': train_dataset.task,
                'metric': train_dataset.metric}, meta_info


def _build_loso_splits(meta_json, seed: int = 0):
    """Leave-One-Site-Out splits.

    Returns all_idx in the same shape as the cached *.index files
    (dict of split-name -> list of folds, each fold = list of subject
    indices into the global G_dataset).

    For each fold f (0 <= f < S):
      * ood_test[f]: every subject from site sites[f] -- the truly
        unseen test domain. Never appears in train.
      * ood_val[f]:  every subject from site sites[(f+1) % S] -- a
        second held-out site, used for early stopping / S_t scoring so the
        ckpt selection signal is itself OOD (not source-leaked).
      * train, id_val, id_test: 80/10/10 stratified split over
        subjects of the remaining S-2 source sites, stratified per
        (site, label) so every source site and both classes appear in
        each split.
    """
    import random
    from collections import defaultdict

    site_ids = list(meta_json['idx2site'])
    labels = list(meta_json['idx2label'])
    n = len(site_ids)
    assert len(labels) == n, 'meta.json idx2label / idx2site length mismatch'

    # Sorted unique site ids -> deterministic fold ordering across runs.
    sites = sorted(set(site_ids))
    S = len(sites)
    assert S >= 3, f'LOSO needs >=3 sites for train/val/test partitioning; got {S}'

    rng = random.Random(int(seed))

    splits = {'train': [], 'id_val': [], 'id_test': [],
              'ood_val': [], 'ood_test': []}

    for f, test_site in enumerate(sites):
        val_site = sites[(f + 1) % S]

        ood_test_idx = [i for i in range(n) if site_ids[i] == test_site]
        ood_val_idx = [i for i in range(n) if site_ids[i] == val_site]

        # Group remaining (source) subjects by (site, label) for stratified split.
        groups = defaultdict(list)
        for i in range(n):
            if site_ids[i] in (test_site, val_site):
                continue
            groups[(site_ids[i], labels[i])].append(i)

        train_idx, id_val_idx, id_test_idx = [], [], []
        for key, indices in groups.items():
            rng.shuffle(indices)
            n_total = len(indices)
            # 10% to id_val and 10% to id_test (each at least 1 if the bucket
            # has >= 2 subjects, else give id_val priority).
            n_id_val = max(1, n_total // 10) if n_total >= 2 else 0
            remaining = n_total - n_id_val
            n_id_test = max(1, remaining // 9) if remaining >= 2 else 0
            id_val_idx.extend(indices[:n_id_val])
            id_test_idx.extend(indices[n_id_val:n_id_val + n_id_test])
            train_idx.extend(indices[n_id_val + n_id_test:])

        splits['train'].append(train_idx)
        splits['id_val'].append(id_val_idx)
        splits['id_test'].append(id_test_idx)
        splits['ood_val'].append(ood_val_idx)
        splits['ood_test'].append(ood_test_idx)

    return splits, S, sites


def get_all_split_idx(name):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 10 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 10 fold have unique test set.
    """
    root_idx_dir = './GOOD/data/good_datasets/{}/'.format(name)
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + 'train.index')):
        print("[!] no split at {}".format(root_idx_dir))
        raise NotImplementedError

    # reading idx from the files
    for section in ['train', 'ood_val', 'ood_test', 'id_val', 'id_test']:
        with open(root_idx_dir + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx




def _shrink_fc(fc_matrix):
    """Apply Ledoit-Wolf shrinkage to a correlation matrix for stable edge estimation.
    
    Shrinks toward the identity: S_shrunk = (1-alpha)*S + alpha*I
    """
    n = fc_matrix.shape[0]
    mu = fc_matrix.trace() / n
    alpha = 0.3  # moderate shrinkage
    target = mu * torch.eye(n, device=fc_matrix.device)
    fc_shrunk = (1 - alpha) * fc_matrix + alpha * target
    fc_shrunk.fill_diagonal_(0.0)
    return fc_shrunk


# Global PCA components, fitted once on all subjects
_pca_components = None
_pca_mean = None
_PCA_DIM = 32


def fit_pca_on_train(G_dataset_train):
    """Fit the global PCA basis on the *training* subjects of one fold.

    Replaces the original 'fit on all subjects' behaviour which silently
    leaked OOD test/val site connectivity into the PCA basis. Called once
    per fold, refits the module-level globals _pca_components and
    _pca_mean. PCA is unsupervised but training-fold-restricted now.
    """
    global _pca_components, _pca_mean
    import numpy as np

    all_fc = []
    for g in G_dataset_train:
        fc = g.ndata['FC_features'].numpy()  # [100, 100]
        all_fc.append(fc)

    all_rows = np.concatenate(all_fc, axis=0)  # [N_train_subjects * 100, 100]

    _pca_mean = all_rows.mean(axis=0)
    centered = all_rows - _pca_mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    _pca_components = Vt[:_PCA_DIM].T  # [100, 32]

    explained = (S[:_PCA_DIM] ** 2).sum() / (S ** 2).sum()
    print(f'#IN#PCA fitted on {len(G_dataset_train)} training subjects: '
          f'{all_rows.shape[1]} -> {_PCA_DIM}, explained variance: {explained:.4f}')


# Backward-compat shim (legacy callers, not used inside this branch).
def fit_pca_on_all(G_dataset):
    fit_pca_on_train(list(G_dataset))


def _apply_pca(fc_rows):
    """Project FC rows [100, 100] -> [100, 32] using fitted PCA."""
    centered = fc_rows - torch.from_numpy(_pca_mean).float()
    projected = centered @ torch.from_numpy(_pca_components).float()
    return projected


def dgl_to_pyg(graph, y, domain):
    # --- Node features: PCA-reduced FC rows ---
    fc = graph.ndata['FC_features'].clone()  # [100, 100]
    x = _apply_pca(fc)  # [100, 32]
    
    # --- Edge construction: soft shrinkage FC, no hard threshold ---
    fc_shrunk = _shrink_fc(fc)
    
    edge_thresh = 0.05
    mask = fc_shrunk.abs() > edge_thresh
    mask.fill_diagonal_(False)
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = fc_shrunk[src, dst].unsqueeze(-1)
    
    data = Data(x=x.float(), edge_index=edge_index, edge_weight=edge_weight,
                y=torch.tensor([y], dtype=torch.long), domain=domain)
    data.env_id = domain
    return data
