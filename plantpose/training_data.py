from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import torch
from matplotlib.image import imread
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .augmentation import GraphAugmentation
from .data import read_graph, sample_paths, split_ids


class TrainingDataset(Dataset, GraphAugmentation):
    def __init__(self, root, family, training, limit=None):
        self.family, self.training = family, training
        self.max_change_light_rate = 0.3
        root = Path(root)
        if family == 'mix':
            split = 'train' if training else 'val'
            sources = [(root / domain / split, repeat if training else 1)
                       for domain, repeat in [('guyot', 10), ('lvd_leaf_src', 10), ('lvd_syn', 1),
                                              ('MIPDB_resample_384', 1), ('toda_camera_resample_384', 20),
                                              ('toda_world_resample_384', 10)]]
        else:
            split = {'root': ('train', 'val_aug'), 'guyot': ('train_aug', 'val_aug'),
                     'synthetic': ('train', 'valid')}[family][not training]
            sources = [(root / split, 1)]
        self.rows = []
        for source, repeat in sources:
            ids, _ = split_ids(source)
            rows = []
            for sample_id in ids:
                annotation, image = sample_paths(source, sample_id, family)
                if not image.is_file():
                    raise FileNotFoundError(image)
                original = source / ('data_np' if family == 'mix' else 'data') / sample_id
                rows.append((annotation, image, str(original.relative_to(root))))
            self.rows.extend(rows * repeat)
        if not self.rows:
            raise ValueError('Empty training/validation split')
        if limit is not None:
            if type(limit) is not int or limit < 1 or limit > len(self.rows):
                raise ValueError('Subset limit must be positive and available in this split')
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        annotation, path, sample_id = self.rows[index]
        nodes, edges = read_graph(annotation, self.family)
        image = imread(path).astype(np.float32)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if self.family == 'mix':
            image = cv2.resize(image, (256, 256))
            if self.training:
                pixels = (nodes * torch.tensor([256, 256])).numpy()
                _, image, points, _ = self._augment_one_sample(image.copy(), pixels)
                nodes = torch.tensor(points, dtype=torch.float32)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5] * image.shape[0], [0.5] * image.shape[0])
        height, width = image.shape[-2:]
        if self.family == 'guyot':
            height, width = height // 2, width // 2
            image = TF.resize(image, [height, width])
        if self.family in ('root', 'guyot') and max(height, width) > 1024:
            scale = 1024 / max(height, width)
            size = [int(height * scale), 1024] if width > height else [1024, int(width * scale)]
            image = TF.resize(image, size)
        nodes, edges = torch.as_tensor(nodes).float(), torch.as_tensor(edges).long().reshape(-1, 2)
        if self.training and self.family in ('guyot', 'mix') and (self.family != 'mix' or len(nodes) >= 5):
            rotated_image, rotated_nodes, rotated_edges, _ = self._rotate(image, nodes, edges)
            graph = nx.Graph()
            graph.add_edges_from(rotated_edges.tolist())
            if len(graph) and nx.is_tree(graph) and (self.family != 'mix' or len(rotated_edges) >= 2):
                image, nodes, edges = rotated_image, rotated_nodes, rotated_edges
        if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 2 or not torch.isfinite(nodes).all():
            raise ValueError('Invalid training nodes: ' + sample_id)
        if edges.numel() == 0 or edges.min() < 0 or edges.max() >= len(nodes):
            raise ValueError('Invalid training edges: ' + sample_id)
        return image.contiguous(), nodes, edges, sample_id
