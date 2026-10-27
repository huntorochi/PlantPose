import json
from pathlib import Path
from types import SimpleNamespace


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in value.items()})


def read_settings(path, dataset, section):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    return dict(values[section], **values['datasets'][dataset].get(section, {}))


def distributed_settings(values, world_size):
    """Eight ranks, one configured local batch each; never rescale the LR/loss."""
    if world_size != 8:
        raise ValueError('Use torchrun --nproc_per_node=8 for one eight-GPU experiment')
    local_batch = values['batch_per_gpu']
    if type(local_batch) is not int or local_batch < 1:
        raise ValueError('batch_per_gpu must be a positive integer')
    return values, local_batch * world_size


def load_config(path, dataset):
    values = dict(dataset=dataset, model=read_settings(path, dataset, 'model'),
                  evaluation=read_settings(path, dataset, 'evaluation'))
    model = values['model']
    config = namespace(dict(MODEL=dict(
        NUM_CLASSES=2,
        ENCODER=dict(HIDDEN_DIM=512, POSITION_EMBEDDING='sine', LR_BACKBONE=2e-5,
                     NUM_FEATURE_LEVELS=4, BACKBONE='resnet50', MASKS=False, DILATION=False),
        DECODER=dict(HIDDEN_DIM=512, NHEADS=8, ENC_LAYERS=4, DEC_LAYERS=4,
                     DIM_FEEDFORWARD=512, DROPOUT=model['dropout'], ACTIVATION='relu',
                     NUM_FEATURE_LEVELS=4, DEC_N_POINTS=4, ENC_N_POINTS=4,
                     TWO_STAGE=False, AUX_LOSS=False, WITH_BOX_REFINE=False,
                     OBJ_TOKEN=model['object_queries'], RLN_TOKEN=1, DUMMY_TOKEN=0, RLN_ATTN=True),
        MATCHER=dict(C_CLASS=3, C_NODE=5))))
    config.dataset = values['dataset']
    config.evaluation = namespace(values['evaluation'])
    return config
