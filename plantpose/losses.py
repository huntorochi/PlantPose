"""Graph losses for normalized (x, y) nodes.

Edge loss: 6U without MST; 3U+3C with MST, sampled independently.
U is unconstrained; C is MST-constrained. Logit penalty: root -100, others -10.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.sparse import find
from scipy.sparse.csgraph import minimum_spanning_tree
from . import box_ops as box_ops_2D
from scipy.optimize import linear_sum_assignment
MIN_TEST = 1e-24

def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    nan = np.isnan(cost_matrix).any()
    if nan:
        cost_matrix[np.isnan(cost_matrix)] = 100000.0
    return linear_sum_assignment(cost_matrix)

class HungarianMatcher(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.cost_nodes = config.MODEL.MATCHER.C_NODE
        self.cost_class = config.MODEL.MATCHER.C_CLASS

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_nodes'].shape[:2]
        out_nodes = outputs['pred_nodes'][..., :2].flatten(0, 1)
        tgt_nodes = torch.cat([v for v in targets['nodes']])
        cost_nodes = torch.cdist(out_nodes, tgt_nodes, p=1)
        tgt_ids = torch.cat([torch.tensor([1] * v.shape[0]).to(out_nodes.device) for v in targets['nodes']])
        cost_class = -outputs['pred_logits'].flatten(0, 1).softmax(-1)[..., tgt_ids]
        C = self.cost_nodes * cost_nodes + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v) for v in targets['nodes']]
        indices = [linear_sum_assignment_with_inf(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class Criterion(nn.Module):

    def __init__(self, model, config, dataset, training_mst):
        super().__init__()
        self.matcher = HungarianMatcher(config)
        self.relation_embed = model.relation_embed
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.rln_token = config.MODEL.DECODER.RLN_TOKEN
        if self.rln_token != 1:
            raise ValueError('The released checkpoints use exactly one relation token.')
        self.losses = config.TRAIN.LOSSES
        self.weight_dict = dict(boxes=config.TRAIN.W_BBOX, **{'class': config.TRAIN.W_CLASS}, cards=config.TRAIN.W_CARD, nodes=config.TRAIN.W_NODE, edges=config.TRAIN.W_EDGE)
        self.training_mst = training_mst
        self.dataset = dataset
        self.epsilon = 100 if dataset == 'root' else 10
        self.unconstrained_weight = self.constrained_weight = 3

    def forward(self, h, out, target):
        indices = self.matcher(out, target)
        losses = {'class': self.loss_class(out['pred_logits'], indices), 'nodes': self.loss_nodes(out['pred_nodes'][..., :2], target['nodes'], indices), 'boxes': self.loss_boxes(out['pred_nodes'], target['nodes'], indices)}
        if self.training_mst:
            constrained = self.mst_edge_loss_numpy if self.dataset == 'mix' else self.mst_edge_loss
            losses['edges'] = constrained(h, target['nodes'], target['edges'], indices)
        else:
            losses['edges'] = self.edge_loss(h, target['nodes'], target['edges'], indices, factor=6)
        losses['cards'] = self.loss_cardinality(out['pred_logits'], indices)
        losses['total'] = sum((losses[k] * self.weight_dict[k] for k in self.losses))
        return losses

    def loss_class(self, outputs, indices):
        weight = torch.tensor([0.2, 0.8]).to(outputs.device)
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.device)
        targets[idx] = 1.0
        loss = F.cross_entropy(outputs.permute(0, 2, 1), targets, weight=weight, reduction='mean')
        return loss

    def loss_nodes(self, outputs, targets, indices):
        num_nodes = sum((len(t) for t in targets))
        idx = self._get_src_permutation_idx(indices)
        pred_nodes = outputs[idx]
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss = F.l1_loss(pred_nodes, target_nodes, reduction='none')
        loss = loss.sum() / num_nodes
        return loss

    def loss_boxes(self, outputs, targets, indices):
        num_boxes = sum((len(t) for t in targets))
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        width_cat = torch.ones((target_boxes.shape[0], 1), device=target_boxes.device) * (3 / 190)
        height_cat = torch.ones((target_boxes.shape[0], 1), device=target_boxes.device) * (3 / 570)
        target_boxes = torch.cat([target_boxes, width_cat, height_cat], dim=-1)
        loss = 1 - torch.diag(box_ops_2D.generalized_box_iou(box_ops_2D.box_cxcywh_to_xyxy(src_boxes), box_ops_2D.box_cxcywh_to_xyxy(target_boxes)))
        loss = loss.sum() / num_boxes
        return loss

    def loss_cardinality(self, outputs, indices):
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.device)
        targets[idx] = 1.0
        tgt_lengths = torch.as_tensor([t.sum() for t in targets], device=outputs.device)
        card_pred = (outputs.argmax(-1) == outputs.shape[-1] - 1).sum(1)
        loss = F.l1_loss(card_pred.float(), tgt_lengths.float(), reduction='sum') / (outputs.shape[0] * outputs.shape[1])
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return (batch_idx, src_idx)

    def edge_loss(self, h, target_nodes, target_edges, indices, factor=1):
        object_token = h[..., :self.obj_token, :]
        relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
        target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for t in target_edges]
        new_target_edges = []
        for t, (_, i) in zip(target_edges, indices):
            tx = t.clone().detach()
            for idx, k in enumerate(i):
                t[tx == k] = idx
            new_target_edges.append(t)
        edge_labels = []
        relation_feature = []
        loss = 0.0
        for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
            rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
            full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
            full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
            full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
            neg_edges = torch.nonzero(torch.triu(full_adj))
            shuffle = np.random.randn(pos_edge.shape[0]) > 0
            to_shuffle = pos_edge[shuffle, :]
            pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
            idx_ = torch.randperm(neg_edges.shape[0])
            neg_edges = neg_edges[idx_, :].to(pos_edge.device)
            shuffle = np.random.randn(neg_edges.shape[0]) > 0
            to_shuffle = neg_edges[shuffle, :]
            neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
            take_neg = neg_edges.shape[0]
            total_edge = pos_edge.shape[0] + neg_edges.shape[0]
            all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
            edge_labels.append(torch.cat((torch.ones(pos_edge.shape[0], dtype=torch.long), torch.zeros(take_neg, dtype=torch.long)), 0))
            relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :], relation_token[batch_id, ...].repeat(total_edge, 1)), 1))
        relation_feature = torch.cat(relation_feature, 0)
        edge_labels = torch.cat(edge_labels, 0).to(h.device)
        relation_pred = self.relation_embed(relation_feature)
        nllloss_func = nn.NLLLoss(reduction='mean')
        relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1)
        if relation_pred_softmax_batch.lt(MIN_TEST).any():
            relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
            relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST
            relation_pred_softmax_batch = relation_pred_softmax_batch + (relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()
        relation_pred_softmax_batch = relation_pred_softmax_batch.log()
        nlloss_batch = nllloss_func(relation_pred_softmax_batch, edge_labels)
        loss = loss + nlloss_batch
        return factor * loss / h.shape[0]

    def mst_edge_loss(self, h, target_nodes, target_edges, indices):
        loss_unconstrained = self.edge_loss(h, target_nodes, target_edges, indices)
        object_token = h[..., :self.obj_token, :]
        relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
        target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for t in target_edges]
        new_target_edges = []
        for t, (_, i) in zip(target_edges, indices):
            tx = t.clone().detach()
            for idx, k in enumerate(i):
                t[tx == k] = idx
            new_target_edges.append(t)
        loss = 0.0
        for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
            rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
            full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
            cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
            full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
            full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
            neg_edges = torch.nonzero(torch.triu(full_adj))
            mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
            inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.0), torch.tensor(0.0))
            zero_upper_triangular = mask_upper_triangular * inverse_full_adj
            pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)
            shuffle = np.random.randn(pos_edge.shape[0]) > 0
            to_shuffle = pos_edge[shuffle, :]
            pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
            idx_ = torch.randperm(neg_edges.shape[0])
            neg_edges = neg_edges[idx_, :].to(h.device)
            shuffle = np.random.randn(neg_edges.shape[0]) > 0
            to_shuffle = neg_edges[shuffle, :]
            neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
            take_neg = neg_edges.shape[0]
            total_edge = pos_edge.shape[0] + neg_edges.shape[0]
            all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
            edge_label_batch = torch.cat((torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device), torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)
            nllloss_func = nn.NLLLoss(reduction='mean')
            mst_edge_label_batch = torch.cat((torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device), torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
            new_mst_edge_label_batch = torch.cat((torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device), torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)
            new_mst_not_edge_label_batch = torch.cat((torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device), torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)
            relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :], relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
            relation_pred_batch = self.relation_embed(relation_feature_batch)
            relation_pred_batch_fake = relation_pred_batch.detach()
            values, ind = torch.max(relation_pred_batch_fake, dim=1)
            mask_ind = ind == 1
            new_pos_edges = all_edges_[mask_ind]
            new_neg_edges = all_edges_[~mask_ind]
            relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
            cost_pred_batch = relation_pred_softmax_batch[:, 0]
            x, y = all_edges_.t()
            cost_adj_batch[x, y] = cost_pred_batch
            cost_adj_batch[y, x] = cost_pred_batch
            cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)
            mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
            mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()
            mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
            mst_adj_batch[mst_adj_batch != 0] = 1
            mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
            mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
            mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
            new_mst_edge_label_batch[mask_pos & ~mask_mst] = 1
            new_mst_not_edge_label_batch[mask_neg & mask_mst] = 1
            new_mst_edge_label_batch_mask = new_mst_edge_label_batch.to(torch.bool)
            new_mst_not_edge_label_batch_mask = new_mst_not_edge_label_batch.to(torch.bool)
            relation_pred_batch_true = relation_pred_batch.clone()
            relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] = -self.epsilon
            relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] = -self.epsilon
            relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)
            if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                mask = relation_pred_softmax_batch_true < MIN_TEST
                relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()
                relation_pred_softmax_batch_true[mask] = MIN_TEST
            relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
            nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
            loss = loss + nlloss_batch
        loss_constrained = loss / h.shape[0]
        return self.unconstrained_weight * loss_unconstrained + self.constrained_weight * loss_constrained

    def mst_edge_loss_numpy(self, h, target_nodes, target_edges, indices):
        loss_unconstrained = self.edge_loss(h, target_nodes, target_edges, indices)
        object_token = h[..., :self.obj_token, :]
        relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
        target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for t in target_edges]
        new_target_edges = []
        for t, (_, i) in zip(target_edges, indices):
            tx = t.clone().detach()
            for idx, k in enumerate(i):
                t[tx == k] = idx
            new_target_edges.append(t)
        loss = 0.0
        nllloss_func = nn.NLLLoss(reduction='mean')
        for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
            rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
            num_nodes = n.shape[0]
            full_adj = torch.ones((num_nodes, num_nodes)) - torch.diag(torch.ones(num_nodes))
            cost_adj_batch = np.zeros((num_nodes, num_nodes))
            full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
            full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
            neg_edges = torch.nonzero(torch.triu(full_adj))
            mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
            inverse_full_adj = full_adj == 0
            zero_upper_triangular = mask_upper_triangular * inverse_full_adj
            pos_edge = torch.nonzero(zero_upper_triangular)
            shuffle = np.random.randn(pos_edge.shape[0]) > 0
            to_shuffle = pos_edge[shuffle, :]
            pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
            idx_ = torch.randperm(neg_edges.shape[0])
            neg_edges = neg_edges[idx_, :]
            shuffle = np.random.randn(neg_edges.shape[0]) > 0
            to_shuffle = neg_edges[shuffle, :]
            neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
            take_neg = neg_edges.shape[0]
            total_edge = pos_edge.shape[0] + neg_edges.shape[0]
            all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0).to(h.device)
            edge_label_batch = torch.cat((torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device), torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)
            new_mst_edge_label_batch_np = np.zeros(total_edge, dtype=float)
            new_mst_not_edge_label_batch_np = np.zeros(total_edge, dtype=float)
            relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :], relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1).clone()
            relation_pred_batch = self.relation_embed(relation_feature_batch)
            relation_pred_batch_fake = relation_pred_batch.detach()
            values, ind = torch.max(relation_pred_batch_fake, dim=1)
            mask_ind = ind == 1
            new_pos_edges = all_edges_[mask_ind].cpu()
            new_neg_edges = all_edges_[~mask_ind].cpu()
            relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
            cost_pred_batch = relation_pred_softmax_batch[:, 0].cpu().numpy()
            all_edges_ = all_edges_.cpu()
            all_edges_np = all_edges_.numpy()
            x, y = all_edges_np.T
            cost_adj_batch[x, y] = cost_pred_batch
            cost_adj_batch[y, x] = cost_pred_batch
            cost_adj_batch = np.triu(cost_adj_batch, k=1)
            mst_adj_batch = minimum_spanning_tree(cost_adj_batch)
            mst_rows, mst_cols, _ = find(mst_adj_batch)
            mst_edges_set = set(zip(mst_rows, mst_cols))
            mst_edges_set.update(set(zip(mst_cols, mst_rows)))
            new_pos_edges_set = set(map(tuple, new_pos_edges.tolist()))
            mask_pos_np = np.array([tuple(edge) in new_pos_edges_set for edge in all_edges_np], dtype=bool)
            mask_mst_np = np.array([tuple(edge) in mst_edges_set for edge in all_edges_np], dtype=bool)
            new_neg_edges_set = set(map(tuple, new_neg_edges.tolist()))
            mask_neg_np = np.array([tuple(edge) in new_neg_edges_set for edge in all_edges_np], dtype=bool)
            new_mst_edge_label_batch_np[mask_pos_np & ~mask_mst_np] = 1
            new_mst_not_edge_label_batch_np[mask_neg_np & mask_mst_np] = 1
            new_mst_edge_label_batch_mask = torch.from_numpy(new_mst_edge_label_batch_np).to(torch.bool).to(h.device)
            new_mst_not_edge_label_batch_mask = torch.from_numpy(new_mst_not_edge_label_batch_np).to(torch.bool).to(h.device)
            relation_pred_batch_true = relation_pred_batch.clone()
            relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] = -self.epsilon
            relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] = -self.epsilon
            relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)
            if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                mask = relation_pred_softmax_batch_true < MIN_TEST
                relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()
                relation_pred_softmax_batch_true[mask] = MIN_TEST
            relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
            nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
            loss = loss + nlloss_batch
        loss_constrained = loss / h.shape[0]
        return self.unconstrained_weight * loss_unconstrained + self.constrained_weight * loss_constrained
