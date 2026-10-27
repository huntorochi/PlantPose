from functools import partial
import numpy as np

def box_iou_2d_np(boxes1, boxes2):
    area1 = box_area_2d_np(boxes1)
    area2 = box_area_2d_np(boxes2)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
    inter = np.clip(x2 - x1, a_min=0, a_max=None) * np.clip(y2 - y1, a_min=0, a_max=None)
    with np.errstate(invalid='raise'):
        try:
            return inter / (area1[:, None] + area2 - inter + 1e-08)
        except RuntimeWarning as e:
            raise

def box_area_2d_np(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

class BBoxEvaluator:

    def __init__(self, classes, iou_fn=box_iou_2d_np, max_detections=2000):
        self.iou_fn = iou_fn
        self.max_detections = max_detections
        self.results_list = []
        self.metrics = [Metric(classes=classes, iou_list=np.arange(0.1, 1.0, 0.1), iou_range=(0.5, 0.95, 0.05), per_class=True, max_detection=(100,))]
        self.iou_thresholds = self.get_unique_iou_thresholds()
        self.iou_mapping = self.get_indices_of_iou_for_each_metric()

    def get_unique_iou_thresholds(self):
        iou_thresholds = [_i for i in self.metrics for _i in i.get_iou_thresholds()]
        iou_thresholds = list(set(iou_thresholds))
        iou_thresholds.sort()
        return iou_thresholds

    def get_indices_of_iou_for_each_metric(self):
        return [[self.iou_thresholds.index(th) for th in m.get_iou_thresholds()] for m in self.metrics]

    def add(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore=None):
        gt_classes = [batch_elem_classes - 1 for batch_elem_classes in gt_classes]
        pred_classes = [batch_elem_classes - 1 for batch_elem_classes in pred_classes]
        if gt_ignore is None:
            n = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n]
        self.results_list.extend(matching_batch(self.iou_fn, self.iou_thresholds, pred_boxes=pred_boxes, pred_classes=pred_classes, pred_scores=pred_scores, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_ignore=gt_ignore, max_detections=self.max_detections))
        return {}

    def eval_single(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore=None):
        gt_classes = [batch_elem_classes - 1 for batch_elem_classes in gt_classes]
        pred_classes = [batch_elem_classes - 1 for batch_elem_classes in pred_classes]
        if gt_ignore is None:
            gt_ignore = np.zeros_like(gt_classes, dtype=bool)
        results = matching_batch(self.iou_fn, self.iou_thresholds, pred_boxes=pred_boxes, pred_classes=pred_classes, pred_scores=pred_scores, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_ignore=gt_ignore, max_detections=self.max_detections)[0]
        metric_scores = {}
        metric_curves = {}
        for metric_idx, metric in enumerate(self.metrics):
            _filter = partial(self.iou_filter, iou_idx=self.iou_mapping[metric_idx])
            iou_filtered_results = _filter(results)
            score, curve = metric([iou_filtered_results])
            if score is not None:
                metric_scores.update(score)
            if curve is not None:
                metric_curves.update(curve)
        return metric_scores

    def eval(self):
        metric_scores = {}
        metric_curves = {}
        for metric_idx, metric in enumerate(self.metrics):
            _filter = partial(self.iou_filter, iou_idx=self.iou_mapping[metric_idx])
            iou_filtered_results = list(map(_filter, self.results_list))
            score, curve = metric(iou_filtered_results)
            if score is not None:
                metric_scores.update(score)
            if curve is not None:
                metric_curves.update(curve)
        return metric_scores

    @staticmethod
    def iou_filter(image_dict, iou_idx, filter_keys=('dtMatches', 'gtMatches', 'dtIgnore')):
        iou_idx = list(iou_idx)
        filtered = {}
        for cls_key, cls_item in image_dict.items():
            filtered[cls_key] = {key: item[iou_idx] if key in filter_keys else item for key, item in cls_item.items()}
        return filtered

    def reset(self):
        self.results_list = []

def matching_batch(iou_fn, iou_thresholds, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore, max_detections):
    results = []
    for pboxes, pclasses, pscores, gboxes, gclasses, gignore in zip(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore):
        img_classes = np.union1d(pclasses, gclasses)
        result = {}
        for c in img_classes:
            pred_mask = pclasses == c
            gt_mask = gclasses == c
            if not np.any(gt_mask):
                result[c] = _matching_no_gt(iou_thresholds=iou_thresholds, pred_scores=pscores[pred_mask], max_detections=max_detections)
            elif not np.any(pred_mask):
                result[c] = _matching_no_pred(iou_thresholds=iou_thresholds, gt_ignore=gignore[gt_mask])
            else:
                result[c] = _matching_single_image_single_class(iou_fn=iou_fn, pred_boxes=pboxes[pred_mask], pred_scores=pscores[pred_mask], gt_boxes=gboxes[gt_mask], gt_ignore=gignore[gt_mask], max_detections=max_detections, iou_thresholds=iou_thresholds)
        results.append(result)
    return results

def _matching_no_gt(iou_thresholds, pred_scores, max_detections):
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]
    dt_scores = pred_scores[dt_ind]
    num_preds = len(dt_scores)
    gt_match = np.array([[]] * len(iou_thresholds))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))
    return {'dtMatches': dt_match, 'gtMatches': gt_match, 'dtScores': dt_scores, 'gtIgnore': np.array([]).reshape(-1), 'dtIgnore': dt_ignore}

def _matching_no_pred(iou_thresholds, gt_ignore):
    dt_scores = np.array([])
    dt_match = np.array([[]] * len(iou_thresholds))
    dt_ignore = np.array([[]] * len(iou_thresholds))
    n_gt = 0 if gt_ignore.size == 0 else gt_ignore.shape[0]
    gt_match = np.zeros((len(iou_thresholds), n_gt))
    return {'dtMatches': dt_match, 'gtMatches': gt_match, 'dtScores': dt_scores, 'gtIgnore': gt_ignore.reshape(-1), 'dtIgnore': dt_ignore}

def _matching_single_image_single_class(iou_fn, pred_boxes, pred_scores, gt_boxes, gt_ignore, max_detections, iou_thresholds):
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]
    pred_boxes = pred_boxes[dt_ind]
    pred_scores = pred_scores[dt_ind]
    gt_ind = np.argsort(gt_ignore, kind='mergesort')
    gt_boxes = gt_boxes[gt_ind]
    gt_ignore = gt_ignore[gt_ind]
    ious = iou_fn(pred_boxes, gt_boxes)
    num_preds, num_gts = (ious.shape[0], ious.shape[1])
    gt_match = np.zeros((len(iou_thresholds), num_gts))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))
    for tind, t in enumerate(iou_thresholds):
        for dind, _d in enumerate(pred_boxes):
            iou = min([t, 1 - 1e-10])
            m = -1
            for gind, _g in enumerate(gt_boxes):
                if gt_match[tind, gind] > 0:
                    continue
                if m > -1 and gt_ignore[m] == 0 and (gt_ignore[gind] == 1):
                    break
                if ious[dind, gind] < iou:
                    continue
                iou = ious[dind, gind]
                m = gind
            if m == -1:
                continue
            else:
                dt_ignore[tind, dind] = int(gt_ignore[m])
                dt_match[tind, dind] = 1
                gt_match[tind, m] = 1
    return {'dtMatches': dt_match, 'gtMatches': gt_match, 'dtScores': pred_scores, 'gtIgnore': gt_ignore.reshape(-1), 'dtIgnore': dt_ignore}

class Metric:

    def __init__(self, classes, iou_list=(0.1, 0.5, 0.75), iou_range=(0.1, 0.5, 0.05), max_detection=(1, 5, 100), per_class=True):
        self.classes = classes
        self.per_class = per_class
        iou_list = np.array(iou_list)
        _iou_range = np.linspace(iou_range[0], iou_range[1], int(np.round((iou_range[1] - iou_range[0]) / iou_range[2])) + 1, endpoint=True)
        self.iou_thresholds = np.union1d(iou_list, _iou_range)
        self.iou_range = iou_range
        self.iou_list_idx = np.nonzero(iou_list[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]
        self.iou_range_idx = np.nonzero(_iou_range[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]
        assert (self.iou_thresholds[self.iou_list_idx] == iou_list).all()
        assert (self.iou_thresholds[self.iou_range_idx] == _iou_range).all()
        self.recall_thresholds = np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) + 1, endpoint=True)
        self.max_detections = max_detection

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def get_iou_thresholds(self):
        return self.iou_thresholds

    def compute(self, results_list):
        dataset_statistics = self.compute_statistics(results_list=results_list)
        results = {}
        results.update(self.compute_ap(dataset_statistics))
        results.update(self.compute_ar(dataset_statistics))
        return (results, None)

    def compute_ap(self, dataset_statistics):
        results = {}
        if self.iou_range:
            key = f'mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_MaxDet_{self.max_detections[-1]}'
            results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idx, max_det_idx=-1)
            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):
                    key = f'{cls_str}_mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_MaxDet_{self.max_detections[-1]}'
                    results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idx, cls_idx=cls_idx, max_det_idx=-1)
        for idx in self.iou_list_idx:
            key = f'AP_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}'
            results[key] = self.select_ap(dataset_statistics, iou_idx=[idx], max_det_idx=-1)
            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):
                    key = f'{cls_str}_AP_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}'
                    results[key] = self.select_ap(dataset_statistics, iou_idx=[idx], cls_idx=cls_idx, max_det_idx=-1)
        return results

    def compute_ar(self, dataset_statistics):
        results = {}
        for max_det_idx, max_det in enumerate(self.max_detections):
            key = f'mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_MaxDet_{max_det}'
            results[key] = self.select_ar(dataset_statistics, max_det_idx=max_det_idx, iou_idx=self.iou_range_idx)
            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):
                    key = f'{cls_str}_mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_MaxDet_{max_det}'
                    results[key] = self.select_ar(dataset_statistics, cls_idx=cls_idx, max_det_idx=max_det_idx, iou_idx=self.iou_range_idx)
        for idx in self.iou_list_idx:
            key = f'AR_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}'
            results[key] = self.select_ar(dataset_statistics, iou_idx=idx, max_det_idx=-1)
            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):
                    key = f'{cls_str}_AR_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}'
                    results[key] = self.select_ar(dataset_statistics, iou_idx=idx, cls_idx=cls_idx, max_det_idx=-1)
        return results

    @staticmethod
    def select_ap(dataset_statistics, iou_idx=None, cls_idx=None, max_det_idx=-1):
        prec = dataset_statistics['precision']
        if iou_idx is not None:
            prec = prec[iou_idx]
        if cls_idx is not None:
            prec = prec[..., cls_idx, :]
        prec = prec[..., max_det_idx]
        return np.mean(prec)

    @staticmethod
    def select_ar(dataset_statistics, iou_idx=None, cls_idx=None, max_det_idx=-1):
        rec = dataset_statistics['recall']
        if iou_idx is not None:
            rec = rec[iou_idx]
        if cls_idx is not None:
            rec = rec[..., cls_idx, :]
        rec = rec[..., max_det_idx]
        if len(rec[rec > -1]) == 0:
            rec = -1
        else:
            rec = np.mean(rec[rec > -1])
        return rec

    def check_number_of_iou(self, *args) -> None:
        num_ious = len(self.get_iou_thresholds())
        for arg in args:
            assert arg.shape[0] == num_ious

    def compute_statistics(self, results_list):
        num_iou_th = len(self.iou_thresholds)
        num_recall_th = len(self.recall_thresholds)
        num_classes = len(self.classes)
        num_max_detections = len(self.max_detections)
        precision = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))
        recall = -np.ones((num_iou_th, num_classes, num_max_detections))
        scores = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))
        for cls_idx, cls_i in enumerate(self.classes):
            for maxDet_idx, maxDet in enumerate(self.max_detections):
                results = [r[cls_idx] for r in results_list if cls_idx in r]
                if len(results) == 0:
                    continue
                dt_scores = np.concatenate([r['dtScores'][0:maxDet] for r in results])
                inds = np.argsort(-dt_scores, kind='mergesort')
                dt_scores_sorted = dt_scores[inds]
                dt_matches = np.concatenate([r['dtMatches'][:, 0:maxDet] for r in results], axis=1)[:, inds]
                dt_ignores = np.concatenate([r['dtIgnore'][:, 0:maxDet] for r in results], axis=1)[:, inds]
                self.check_number_of_iou(dt_matches, dt_ignores)
                gt_ignore = np.concatenate([r['gtIgnore'] for r in results])
                num_gt = np.count_nonzero(gt_ignore == 0)
                if num_gt == 0:
                    continue
                tps = np.logical_and(dt_matches, np.logical_not(dt_ignores))
                fps = np.logical_and(np.logical_not(dt_matches), np.logical_not(dt_ignores))
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)
                for th_ind, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp, fp = (np.array(tp), np.array(fp))
                    r, p, s = compute_stats_single_threshold(tp, fp, dt_scores_sorted, self.recall_thresholds, num_gt)
                    recall[th_ind, cls_idx, maxDet_idx] = r
                    precision[th_ind, :, cls_idx, maxDet_idx] = p
                    scores[th_ind, :, cls_idx, maxDet_idx] = s
        return {'counts': [num_iou_th, num_recall_th, num_classes, num_max_detections], 'recall': recall, 'precision': precision, 'scores': scores}

def compute_stats_single_threshold(tp, fp, dt_scores_sorted, recall_thresholds, num_gt):
    num_recall_th = len(recall_thresholds)
    rc = tp / num_gt
    pr = tp / (fp + tp + np.spacing(1))
    if len(tp):
        recall = rc[-1]
    else:
        recall = 0
    precision = np.zeros((num_recall_th,))
    th_scores = np.zeros((num_recall_th,))
    pr = pr.tolist()
    precision = precision.tolist()
    for i in range(len(tp) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    inds = np.searchsorted(rc, recall_thresholds, side='left')
    try:
        for save_idx, array_index in enumerate(inds):
            precision[save_idx] = pr[array_index]
            th_scores[save_idx] = dt_scores_sorted[array_index]
    except:
        pass
    return (recall, np.array(precision), np.array(th_scores))
