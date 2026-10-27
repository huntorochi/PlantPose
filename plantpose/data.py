"""Image and graph datasets."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.image import imread
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

MIX_DOMAINS = dict(guyot='guyot', leaf='lvd_leaf_src', MIPDB='MIPDB_resample_384',
                   mtree='mtree_60000', toda_cam='toda_camera_resample_384',
                   toda_world='toda_world_resample_384', leaf_mask='lvd_syn')
HELD_OUT_DOMAINS = ('lsystem', 'bkg_leaf', 'mtree_500', 'root_ood')


def split_ids(root):
    metadata = json.loads((Path(root) / 'ids.json').read_text(encoding='utf8'))
    ids = metadata['ids']
    if not ids or len(set(ids)) != len(ids) or any(Path(n).name != n or '/' in n or '\\' in n for n in ids):
        raise ValueError('Invalid or duplicate sample IDs: ' + str(root))
    return ids, {k: v for k, v in metadata.items() if k != 'ids'}


def sample_paths(root, sample_id, family):
    stem = Path(sample_id).stem
    image_stem = stem[len('DETR_all_'):] if family == 'synthetic' else stem
    return Path(root) / 'data' / (stem + '.npz'), Path(root) / 'rgb' / (image_stem + '.png')


def read_graph(path, family):
    with np.load(path, allow_pickle=False) as row:
        nodes, edges = torch.from_numpy(row['nodes']), torch.from_numpy(row['edges'])
    if family == 'mix':
        nodes = nodes.clamp(1e-6, 1-1e-6)
    return nodes, edges


class GraphDataset(Dataset):
    """Read one test split. Nodes are normalized (x,y); edge order is preserved."""
    def __init__(self, root, dataset, domain=None):
        self.root, self.dataset, self.domain = Path(root), dataset, domain
        self.held_out = dataset == 'mix' and domain in HELD_OUT_DOMAINS
        if dataset == 'mix':
            if domain not in MIX_DOMAINS and not self.held_out:
                raise ValueError('mix requires one of: ' + ', '.join((*MIX_DOMAINS, *HELD_OUT_DOMAINS)))
            if not self.held_out:
                self.root = self.root / MIX_DOMAINS[domain] / 'test'
        self.ids, self.split_info = split_ids(self.root)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        annotation, image_path = sample_paths(self.root, sample_id, self.dataset)
        nodes, edges = read_graph(annotation, self.dataset)
        image = imread(image_path).astype(np.float32)
        if self.held_out and self.domain in ('bkg_leaf', 'mtree_500') and image.max() > 1:
            image = image / 255.0
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if self.dataset == 'mix' and not self.held_out:
            image = cv2.resize(image, (256, 256))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5] * image.shape[0], [0.5] * image.shape[0])
        if self.held_out:
            # Resize held-out images after normalization.
            image = TF.resize(image, [256, 256])
        height, width = image.shape[-2:]
        if self.dataset == 'guyot':
            height, width = height // 2, width // 2
            image = TF.resize(image, [height, width])
        if self.dataset in ('root', 'guyot'):
            maximum = 1024 if self.dataset == 'root' else 2048
            if max(height, width) > maximum:
                scale = maximum / max(height, width)
                size = [int(height * scale), maximum] if width > height else [maximum, int(width * scale)]
                image = TF.resize(image, size)
        nodes, edges = nodes.float(), edges.long().reshape(-1, 2)
        if nodes.ndim != 2 or nodes.shape[1] != 2 or not torch.isfinite(nodes).all():
            raise ValueError(f'Invalid nodes in {sample_id}')
        if edges.numel() and (edges.min() < 0 or edges.max() >= len(nodes)):
            raise ValueError(f'Invalid edge index in {sample_id}')
        return image.contiguous(), nodes, edges, sample_id


def collate_graphs(batch):
    images, nodes, edges, ids = map(list, zip(*batch))
    # Noise augmentation can produce float64 images.
    return [image.to(torch.float32) for image in images], nodes, edges, ids
