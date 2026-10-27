"""Train one PlantPose experiment with torchrun --nproc_per_node=8."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'configs/plantpose.json'))
    parser.add_argument('--dataset', required=True, choices=['root', 'guyot', 'synthetic', 'mix'])
    parser.add_argument('--data', required=True, help='parent of the train/validation splits')
    parser.add_argument('--output', required=True, help='new directory for checkpoints and epoch summaries')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--training-mst', action='store_true', dest='training_mst')
    group.add_argument('--no-training-mst', action='store_false', dest='training_mst')
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--backbone-weights', help='ImageNet ResNet50 weights for a new run')
    weights.add_argument('--resume', help='<dataset>_mst.pt or <dataset>_unmst.pt; restores optimizer and RNG')
    parser.add_argument('--epochs', type=int, help='final epoch, including restored epochs; defaults to config')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 8:
        parser.error('Use torchrun --standalone --nproc_per_node=8 train.py ...')
    if args.workers < 0 or args.epochs is not None and args.epochs < 1:
        parser.error('workers must be nonnegative and epochs positive')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['MPLCONFIGDIR'] = str(Path(args.output).resolve().parent / '.cache/matplotlib')
    from plantpose.distributed_training import train_distributed
    train_distributed(args)


if __name__ == '__main__':
    main()
