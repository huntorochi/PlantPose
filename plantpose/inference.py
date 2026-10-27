"""Graph decoding with optional minimum spanning trees."""
import itertools
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


def kruskal_edges(pairs, costs):
    graph = nx.Graph()
    graph.add_weighted_edges_from((int(u), int(v), float(w))
                                 for (u, v), w in zip(pairs.numpy(), costs.cpu().numpy()))
    adjacency = np.zeros((len(graph), len(graph)))
    for u, v in nx.minimum_spanning_edges(graph, algorithm='kruskal', data=False):
        adjacency[u, v] = adjacency[v, u] = graph[u][v]['weight']
    return torch.nonzero(torch.tensor(adjacency) * torch.triu(torch.ones_like(torch.tensor(adjacency)), diagonal=1))


def prim_edges(pairs, costs, count):
    adjacency = torch.zeros((count, count), device=costs.device)
    x, y = pairs.t()
    adjacency[x, y] = adjacency[y, x] = costs
    selected = torch.zeros(count, device=costs.device, dtype=torch.bool)
    result = torch.zeros_like(adjacency)
    selected[0] = True
    while not selected.all():
        mask = selected.unsqueeze(1) & ~selected.unsqueeze(0) & (adjacency > 0)
        candidates = adjacency.clone()
        candidates[~mask] = float('inf')
        minimum, endpoints = torch.min(candidates, dim=1)
        weight, start = torch.min(minimum, dim=0)
        if not torch.isfinite(weight):
            raise ValueError('Prim decoder cannot connect the remaining nodes with positive costs.')
        end = endpoints[start]
        selected[end] = True
        result[start, end] = result[end, start] = adjacency[start, end]
    # Emit both directions for each selected edge.
    return torch.nonzero(result)


@torch.no_grad()
def decode(h, out, model, object_queries, test_mst, algorithm='kruskal'):
    objects, relation = h[..., :object_queries, :], h[..., object_queries:object_queries+1, :]
    valid = out['pred_logits'].argmax(-1).detach()
    predictions = []
    for batch in range(h.shape[0]):
        ids = torch.nonzero(valid[batch]).squeeze(1)
        nodes = out['pred_nodes'][batch, ids, :2].detach()
        boxes = out['pred_nodes'][batch, ids, :].detach().cpu().numpy()
        scores = out['pred_logits'].softmax(-1)[batch, ids, 1].detach().cpu().numpy()
        classes = valid[batch, ids].cpu().numpy()
        if len(ids) > 1:
            original_pairs = list(map(list, zip(*itertools.combinations(list(ids), 2))))
            pairs = torch.tensor(list(itertools.combinations(range(len(ids)), 2)))
            repeated_relation = relation[batch].repeat(len(pairs), 1)
            first = torch.cat((objects[batch, original_pairs[0]], objects[batch, original_pairs[1]], repeated_relation), 1)
            second = torch.cat((objects[batch, original_pairs[1]], objects[batch, original_pairs[0]], repeated_relation), 1)
            logits = (model.relation_embed(first).detach() + model.relation_embed(second).detach()) / 2.0
            if test_mst:
                costs = F.softmax(logits, dim=-1).detach()[:, 0]
                edges = prim_edges(pairs, costs, len(ids)) if algorithm == 'prim' else kruskal_edges(pairs, costs)
                lookup = {tuple(p): i for i, p in enumerate(pairs.tolist())}
                selected = np.array([lookup[tuple(sorted((int(p[0]), int(p[1]))))] for p in edges if p[0] != p[1]], dtype=np.int64)
                edges = edges.cpu().numpy()
            else:
                selected = torch.nonzero(logits.argmax(-1)).squeeze(1).cpu().numpy()
                edges = pairs[selected].cpu().numpy()
            edge_scores = logits.softmax(-1)[selected, 1].cpu().numpy()
            edge_classes = logits.argmax(-1)[selected].cpu().numpy()
        else:
            edges = np.empty((0, 2), dtype=np.int64)
            edge_scores, edge_classes = np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
        predictions.append((nodes, edges, boxes, scores, classes, edge_scores, edge_classes))
    return tuple(map(list, zip(*predictions)))
