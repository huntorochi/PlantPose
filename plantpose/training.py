import os
from types import SimpleNamespace

import torch

from .config import load_config, read_settings
from .inference import decode
from .metrics.smd import street_mover_distance


def settings(path, dataset, training_mst=False):
    values = dict(read_settings(path, dataset, 'training'), dataset=dataset)
    if training_mst and 'mst_lr_drop' in values:
        values['lr_drop'] = values['mst_lr_drop']
    config = load_config(path, dataset)
    config.TRAIN = SimpleNamespace(LOSSES=['boxes', 'class', 'cards', 'nodes', 'edges'],
                                  W_BBOX=2., W_CLASS=3., W_CARD=1., W_NODE=5., W_EDGE=values['edge_weight'])
    return config, values


def optimizer_for(model, values):
    ordinary, backbone, offsets = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = backbone if 'encoder.0' in name else offsets if any(k in name for k in ('reference_points', 'sampling_offsets')) else ordinary
        target.append(parameter)
    groups = [dict(params=ordinary, lr=values['learning_rate']),
              dict(params=backbone, lr=values['backbone_learning_rate']),
              dict(params=offsets, lr=values['learning_rate'] * 0.1)]
    optimizer = torch.optim.AdamW(groups, lr=values['learning_rate'], weight_decay=values['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, values['lr_drop'], gamma=values['lr_gamma'])
    return optimizer, scheduler


def initialize_backbone(model, path):
    state = torch.load(path, map_location='cpu')
    state = {k: v for k, v in state.items() if not k.startswith('fc.')}
    model.encoder[0].body.load_state_dict(state, strict=True)


def device_batch(batch, device):
    images, nodes, edges, ids = batch
    return [x.to(device) for x in images], dict(nodes=[x.to(device) for x in nodes],
                                              edges=[x.to(device) for x in edges]), ids


@torch.no_grad()
def validation_value(h, out, model, criterion, target, config, values, device):
    predictions = decode(h, out, model, config.MODEL.DECODER.OBJ_TOKEN,
                         criterion.training_mst, values['validation_mst_algorithm'])
    distances = [street_mover_distance(n, e, pn, pe, device, values['smd_points'],
                                       values['smd_eps'], values['smd_max_iter'])
                 for n, e, pn, pe in zip(target['nodes'], target['edges'], predictions[0], predictions[1])]
    return float(torch.tensor(distances).sum() if values['validation'] == 'smd_batch_sum'
                 else torch.tensor(distances).mean())


def save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_checkpoints(output, dataset, training_mst, state, improved):
    name = dataset + ('_mst' if training_mst else '_unmst')
    latest = output / (name + '.pt')
    save(latest, state)
    if improved:
        save(output / (name + '_best.pt'), state)
    return latest
