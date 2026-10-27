"""Evaluate a released checkpoint: python evaluate.py --help."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'configs/plantpose.json'))
    parser.add_argument('--dataset', required=True, choices=['root', 'guyot', 'synthetic', 'mix'])
    parser.add_argument('--data', required=True, help='released data/rgb split; in-domain mix uses its domain parent')
    parser.add_argument('--checkpoint', required=True, help='checkpoints/<dataset>_mst.pt or <dataset>_unmst.pt')
    parser.add_argument('--output', required=True, help='new output directory')
    parser.add_argument('--domain', choices=['guyot', 'leaf', 'leaf_mask', 'MIPDB', 'mtree', 'toda_cam', 'toda_world',
                                           'lsystem', 'bkg_leaf', 'mtree_500', 'root_ood'])
    for name in ('training-mst', 'test-mst'):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--' + name, dest=name.replace('-', '_'), action='store_true')
        group.add_argument('--no-' + name, dest=name.replace('-', '_'), action='store_false')
    parser.add_argument('--gpu', default='0', help='physical GPU index, or cpu')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int, help='optional quick check; omit to evaluate the complete split')
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit is not None and args.limit < 1:
        parser.error('batch-size and limit must be positive')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.gpu == 'cpu' else args.gpu
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cache = Path(args.output).resolve().parent / '.cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')
    dll_handles = []
    if os.name == 'nt':
        for folder in (Path(sys.prefix), Path(sys.prefix)/'Library/bin', Path(sys.prefix)/'DLLs'):
            if folder.is_dir():
                dll_handles.append(os.add_dll_directory(str(folder)))
        os.environ['PATH'] = str(Path(sys.prefix)/'Library/bin') + os.pathsep + os.environ['PATH']
    from plantpose.evaluation import evaluate
    evaluate(args)


if __name__ == '__main__':
    main()
