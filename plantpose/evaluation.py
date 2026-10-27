import hashlib
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .box_ops import box_cxcywh_to_xyxy_np
from .config import load_config
from .data import GraphDataset, collate_graphs
from .inference import decode
from .models import build_relationformer
from .metrics.ap import BBoxEvaluator
from .metrics.keypoints import keypoint_graph
from .metrics.smd import street_mover_distance
from .metrics.topology import compute_topo


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('net', checkpoint)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def edge_boxes(nodes, edges):
    nodes = nodes.detach().cpu().numpy()
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    boxes = np.hstack((nodes[edges[:, 0]], nodes[edges[:, 1]]))
    return np.hstack((np.minimum(boxes[:, :2], boxes[:, 2:]), np.maximum(boxes[:, :2], boxes[:, 2:])))


def fixed_boxes(nodes, width):
    return np.concatenate([nodes.cpu().numpy(), np.ones_like(nodes.cpu()) * width], axis=1)


class Scores:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.ap = {k: BBoxEvaluator([k], max_detections=config.max_detections)
                   for k in ('node', 'edge', 'node_keypoint', 'edge_keypoint')}
        self.smd, self.topology = {'full': [], 'keypoint': []}, {'full': [], 'keypoint': []}
        self.samples, self.trees = 0, 0

    def add_graph(self, kind, nodes, edges, prediction):
        pn, pe, pb, ps, pc, es, ec = prediction
        suffix = '' if kind == 'full' else '_keypoint'
        self.ap['node' + suffix].add(
            pred_boxes=[box_cxcywh_to_xyxy_np(pb)], pred_classes=[pc], pred_scores=[ps],
            gt_boxes=[box_cxcywh_to_xyxy_np(fixed_boxes(nodes, self.config.node_box_size))],
            gt_classes=[np.ones(len(nodes))])
        self.ap['edge' + suffix].add(
            pred_boxes=[edge_boxes(pn, pe)], pred_classes=[ec], pred_scores=[es],
            gt_boxes=[edge_boxes(nodes, edges)], gt_classes=[np.ones(len(edges))])
        self.smd[kind].append(street_mover_distance(nodes, edges, pn, pe, self.device,
                             self.config.smd_points, self.config.smd_eps, self.config.smd_max_iter))
        self.topology[kind].append(compute_topo(nodes.cpu(), edges.cpu(), pn.cpu(), pe,
                                              img_size=self.config.topo_image_size))

    def add(self, nodes, edges, predictions, synthetic=False):
        for i in range(len(nodes)):
            self.samples += 1
            pe = predictions[1][i]
            if len(pe):
                graph = nx.Graph()
                graph.add_edges_from(pe.tolist())
                self.trees += int(nx.is_tree(graph))
        reduced = keypoint_graph(nodes, edges, *predictions, synthetic=synthetic)
        for i in range(len(nodes)):
            self.add_graph('full', nodes[i], edges[i], [p[i] for p in predictions])
            self.add_graph('keypoint', reduced[0][i], reduced[1][i], [p[i] for p in reduced[2:]])

    def result(self):
        result = {'samples': self.samples, 'trees': self.trees,
                  'AP_AR': {k: v.eval() for k, v in self.ap.items()}}
        for kind in ('full', 'keypoint'):
            values = torch.tensor(self.smd[kind])
            p, r = np.array(self.topology[kind]).mean(axis=0)
            result[kind] = dict(smd_mean=values.mean().item(), smd_std=values.std().item(),
                                precision=float(p), recall=float(r), f1=float(2*p*r/(p+r)) if p+r else 0.)
        return result


def evaluate(args):
    start = time.monotonic()
    config = load_config(args.config, args.dataset)
    if args.training_mst and not args.test_mst:
        raise ValueError('This release validates three conditions; MST training requires MST testing.')
    device = torch.device('cpu' if args.gpu == 'cpu' else 'cuda:0')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    dataset = GraphDataset(args.data, config.dataset, args.domain)
    split_info = dataset.split_info
    ids = dataset.ids[:args.limit] if args.limit else dataset.ids
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    model = build_relationformer(config).to(device)
    load_weights(model, args.checkpoint)
    model.eval()
    scores = Scores(config.evaluation, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_graphs, shuffle=False, num_workers=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'graphs.jsonl').open('x', encoding='utf-8') as graphs, torch.no_grad():
        for images, nodes, edges, batch_ids in loader:
            images = [image.to(device) for image in images]
            h, out = model(images)
            predictions = decode(h, out, model, config.MODEL.DECODER.OBJ_TOKEN,
                                 args.test_mst, config.evaluation.mst_algorithm)
            if config.evaluation.fixed_prediction_box:
                predictions[2][:] = [fixed_boxes(n, config.evaluation.node_box_size) for n in predictions[0]]
            for i, sample_id in enumerate(batch_ids):
                pn, pe, pb, ps, pc, es, ec = [p[i] for p in predictions]
                graphs.write(json.dumps(dict(id=sample_id, nodes=pn.cpu().tolist(), edges=pe.tolist(),
                    boxes=pb.tolist(), node_scores=ps.tolist(), node_classes=pc.tolist(),
                    edge_scores=es.tolist(), edge_classes=ec.tolist()), separators=(',', ':')) + '\n')
            scores.add(nodes, edges, predictions, synthetic=config.dataset == 'synthetic')
            if scores.samples % 25 == 0 or scores.samples == len(dataset):
                seconds = time.monotonic() - start
                print(f'{scores.samples}/{len(dataset)} samples, ETA {(len(dataset)-scores.samples)*seconds/max(scores.samples,1)/60:.1f} min', flush=True)
    result = scores.result()
    result.update(dataset=config.dataset, domain=args.domain, split=split_info, training_mst=args.training_mst,
                  test_mst=args.test_mst, seed=args.seed, benchmark=torch.backends.cudnn.benchmark,
                  checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.config),
                  sample_ids=ids, full_split=args.limit is None, seconds=time.monotonic()-start)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('samples', 'full', 'keypoint', 'seconds')}, indent=2))
    return result
