"""Prepare images and existing graph labels. Outputs must be new directories."""
import argparse
import csv
import json
import random
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import cv2
import numpy as np
from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

SPLITS = json.loads(Path(__file__).with_name('splits.json').read_text(encoding='utf-8'))


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return image


def read_graph(path):
    with np.load(path, allow_pickle=False) as data:
        return data['nodes'].copy(), data['edges'].copy()


def validate(nodes, edges):
    nodes = np.asarray(nodes, dtype=np.float32)
    edges = np.asarray(edges)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 2:
        raise ValueError('Expected at least two (x,y) nodes')
    if not np.isfinite(nodes).all() or (nodes < 0).any() or (nodes > 1).any():
        raise ValueError('Nodes must lie in [0,1]; no automatic clipping')
    if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges):
        raise ValueError('Expected nonempty (E,2) edges')
    if not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
        raise ValueError('Edge indices must be integers')
    edges = edges.astype(np.int64)
    if edges.min() < 0 or edges.max() >= len(nodes):
        raise ValueError('Invalid edge index')
    return nodes, edges


def save_sample(root, sample_id, image, nodes, edges):
    root = Path(root)
    if '/' in sample_id or '\\' in sample_id or Path(sample_id).name != sample_id:
        raise ValueError('Expected a filename-only sample ID')
    stem = Path(sample_id).stem
    if stem in ('', '.', '..'):
        raise ValueError('Invalid sample ID')
    nodes, edges = validate(nodes, edges)
    png, npz = root/'rgb'/(stem+'.png'), root/'data'/(stem+'.npz')
    if png.exists() or npz.exists():
        raise FileExistsError('Refusing to overwrite sample: ' + str(root/stem))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 BGR image')
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('PNG encoding failed')
    png.parent.mkdir(parents=True, exist_ok=True)
    npz.parent.mkdir(parents=True, exist_ok=True)
    with png.open('xb') as f:
        f.write(encoded.tobytes())
    with npz.open('xb') as f:
        np.savez_compressed(f, nodes=nodes, edges=edges)


def save_ids(root, ids, family, preview=False):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate IDs')
    with (Path(root)/'ids.json').open('x', encoding='utf-8') as f:
        json.dump(dict(ids=ids, family=family, samples=len(ids),
                       selection='preview' if preview else 'fixed_ids'), f)


def image_index(root):
    result = {}
    for p in sorted(Path(root).rglob('*')):
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        if p.stem in result:
            raise ValueError('Ambiguous image stem: ' + p.stem)
        result[p.stem] = p
    return result


def from_annotation(annotation, width, height):
    if not isinstance(annotation, dict):
        with open(annotation, encoding='utf-8') as f:
            annotation = json.load(f)
    features = annotation['VineImage'][0]['VineFeature'][0]
    count = len(features)
    if sorted(row['FeatureID'] for row in features) != list(range(count)):
        raise ValueError('Expected contiguous official FeatureID values')
    nodes = np.zeros((count, 2), dtype=np.float32)
    adjacency = np.zeros((count, count), dtype=bool)
    for row in features:
        index = row['FeatureID']
        parent = 0 if row['ParentID'] is None else row['ParentID']
        if not isinstance(parent, int) or not 0 <= parent < count:
            raise ValueError('Invalid ParentID')
        nodes[index] = row['FeatureCoordinates']
        adjacency[index, parent] = adjacency[parent, index] = True
    edges = np.argwhere(np.triu(adjacency, 1)).astype(np.int64)
    return nodes/np.asarray([width, height], dtype=np.float32), edges


def crop_resize(image, row):
    expected = int(row['source_height']), int(row['source_width'])
    if image.shape[:2] != expected:
        raise ValueError('Source dimensions differ for ' + row['sample_id'])
    x, y, w, h = [int(row[k]) for k in ('crop_x', 'crop_y', 'crop_width', 'crop_height')]
    if min(x, y) < 0 or min(w, h) < 1 or x+w > image.shape[1] or y+h > image.shape[0]:
        raise ValueError('Invalid crop for ' + row['sample_id'])
    return cv2.resize(image[y:y+h, x:x+w], (int(row['output_width']), int(row['output_height'])), interpolation=cv2.INTER_LINEAR)


def composite(image, mask, background, nodes, edges):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError('Image/mask dimensions differ')
    height, width = image.shape[:2]
    gray = rgb2gray(mask) if mask.ndim == 3 else mask
    binary = gray < threshold_otsu(gray)
    foreground = np.column_stack(np.where(binary))
    if len(foreground) < 2:
        raise ValueError('Empty foreground mask')
    obj_h, obj_w = foreground.max(0)-foreground.min(0)
    if min(obj_h, obj_w) <= 0:
        raise ValueError('Degenerate foreground mask')
    sy = random.uniform(0.5, 0.99*height/obj_h)
    sx = random.uniform(0.5, 0.99*width/obj_w)
    nh, nw = int(height*sy), int(width*sx)
    foreground_image = cv2.resize(image, (nw, nh))
    foreground_mask = cv2.resize(binary.astype(np.uint8), (nw, nh))
    positions = np.column_stack(np.where(foreground_mask))
    ymin, xmin = positions.min(0)
    ymax, xmax = positions.max(0)
    dx = np.random.randint(-xmin, max(width-xmax, -xmin+1))
    dy = np.random.randint(-ymin, max(height-ymax, -ymin+1))
    y0, y1, x0, x1 = max(0, dy), min(height, dy+nh), max(0, dx), min(width, dx+nw)
    fy, fx = max(0, -dy), max(0, -dx)
    fg = foreground_image[fy:fy+y1-y0, fx:fx+x1-x0]
    selection = foreground_mask[fy:fy+y1-y0, fx:fx+x1-x0] == 1
    result = cv2.resize(background, (width, height))
    result[y0:y1, x0:x1][selection] = fg[selection]
    points = np.asarray(nodes, dtype=np.float32)*np.asarray([sx, sy], dtype=np.float32)
    points += np.asarray([dx/width, dy/height], dtype=np.float32)
    return result, points, np.asarray(edges).copy()


def augment(image, nodes):
    """Preserve the original gamma/noise/flip probabilities and RNG draw order."""
    height, width = image.shape[:2]
    a = random.random()
    c = random.random() if a >= 0.3 else None
    if a < 0.2 or (c is not None and (c < 0.8 or c >= 0.9)):
        image = exposure.adjust_gamma(image, random.uniform(0.7, 1.3))
    if 0.2 <= a < 0.3 or (c is not None and c >= 0.8):
        pixels = image.astype(np.float32) / 255
        image = np.uint8(np.clip(pixels + np.random.normal(0, 0.1, pixels.shape), 0, 1) * 255)
    points = np.asarray(nodes, dtype=np.float64).copy()
    if c is not None:
        image = cv2.flip(image, 1)
        points[:, 0] = width - points[:, 0]
    return image, points / [width, height]


def load_sample(root, sample_id):
    stem = Path(sample_id).stem
    return (read_image(root/'rgb'/(stem+'.png')),
            *read_graph(root/'data'/(stem+'.npz')))


def prepare(args):
    preview = args.limit is not None
    if args.command in ('augment', 'background'):
        ids = json.loads((args.source/'ids.json').read_text(encoding='utf-8'))['ids'][:args.limit]
        if args.command == 'background':
            masks = image_index(args.masks)
            backgrounds = list(image_index(args.backgrounds).values())
            if not backgrounds:
                raise ValueError('No local backgrounds')
        output_ids = []
        for sid in ids:
            image, nodes, edges = load_sample(args.source, sid)
            for index in range(args.factor if args.command == 'augment' else 1):
                if args.command == 'augment':
                    result, points = augment(image, nodes*np.asarray(image.shape[1::-1], dtype=np.float32))
                else:
                    mask = cv2.resize(read_image(masks[Path(sid).stem]), image.shape[1::-1])
                    bg = read_image(backgrounds[np.random.randint(len(backgrounds))])
                    result, points, _ = composite(image, mask, bg, nodes, edges)
                name = Path(sid).stem+'_a'+str(index)+'.npz'
                save_sample(args.output, name, result, points, edges)
                output_ids.append(name)
        save_ids(args.output, output_ids, 'guyot' if args.command == 'augment' else 'mix',
                 preview or args.command == 'background')
        return
    if args.command == 'guyot':
        images = image_index(args.source/'01-TrainAndValidationSet')
        test_images = image_index(args.source/'02-IndependentTestSet')
        if images.keys() & test_images.keys():
            raise ValueError('Duplicate official sample IDs')
        images.update(test_images)
    elif args.command == 'web':
        images = image_index(args.images)
        with (args.source/'sources.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = {row['sample_id']: row for row in csv.DictReader(f)}
    dataset = 'mix/toda_world_resample_384' if args.command == 'web' else args.dataset
    for split, fixed_ids in SPLITS[dataset].items():
        if args.split != 'all' and args.split != split:
            continue
        ids = fixed_ids[:args.limit]
        for sid in ids:
            stem = Path(sid).stem
            if args.command == 'guyot':
                path = images[stem]
                image = read_image(path)
                height, width = image.shape[:2]
                nodes, edges = from_annotation(path.with_name(stem+'_annotation.json'), width, height)
                image = cv2.resize(image, (width//4, height//4), interpolation=cv2.INTER_CUBIC)
                if dataset == 'mix/guyot':
                    image = cv2.resize(image, (512, 512))
            elif args.command == 'web':
                row = rows[stem]
                if row['split'] != split:
                    raise ValueError('CSV/split mismatch: '+stem)
                image = crop_resize(read_image(images[row['source_id']]), row)
                nodes, edges = read_graph(args.source/split/'data'/(stem+'.npz'))
            else:
                image, nodes, edges = load_sample(args.source, sid)
            save_sample(args.output/split, sid, image, nodes, edges)
        save_ids(args.output/split, ids, dataset.split('/')[0], preview)


def download(args):
    with args.csv.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    sources = {}
    for row in rows:
        key = row.get('source_id') or row.get('background_id')
        url = row['source_url']
        if not key or Path(key).name != key or '/' in key or '\\' in key or key in ('.', '..'):
            raise ValueError('Invalid source ID')
        if urlparse(url).scheme not in ('http', 'https'):
            raise ValueError('Expected an HTTP(S) source URL')
        if key in sources and sources[key] != url:
            raise ValueError('Conflicting URLs for '+key)
        sources[key] = url
    args.output.mkdir(parents=True)
    failures = []
    items = list(sources.items())[:args.limit]
    for key, url in items:
        try:
            with urlopen(Request(url, headers={'User-Agent':'Mozilla/5.0'}), timeout=30) as response:
                payload = response.read(25*1024*1024+1)
            if len(payload) > 25*1024*1024:
                raise ValueError('Image exceeds 25 MiB limit')
            if cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
                raise ValueError('Response is not an image')
            with (args.output/(key+'.jpg')).open('xb') as f:
                f.write(payload)
        except Exception as exc:
            failures.append({'source_id':key, 'url':url, 'error':str(exc)})
    (args.output/'unavailable.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    print('Downloaded:', len(items)-len(failures), 'Unavailable:', len(failures))
    return bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('guyot', 'split', 'augment', 'web', 'background', 'download'):
        p = subs.add_parser(name)
        p.add_argument('--output', required=True, type=Path)
        p.add_argument('--limit', type=int, help='Preview only; per split where applicable')
        if name == 'download':
            p.add_argument('--csv', required=True, type=Path)
            continue
        p.add_argument('--source', required=True, type=Path)
        p.add_argument('--seed', type=int, default=10)
        if name in ('guyot', 'split'):
            p.add_argument('--dataset', choices=['guyot', 'mix/guyot'] if name == 'guyot' else list(SPLITS), required=True)
        if name in ('guyot', 'split', 'web'):
            p.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all')
        if name == 'web':
            p.add_argument('--images', required=True, type=Path)
        if name == 'augment':
            p.add_argument('--factor', type=int, choices=[32, 100], required=True)
        if name == 'background':
            p.add_argument('--masks', required=True, type=Path)
            p.add_argument('--backgrounds', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.limit is not None and args.limit < 1):
        parser.error('Use a new output directory and a positive limit')
    if args.command == 'download':
        raise SystemExit(download(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    prepare(args)
    print('Prepared:', args.output)


if __name__ == '__main__':
    main()
