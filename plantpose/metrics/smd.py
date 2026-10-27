import math
import numpy as np
import torch
from torch import nn

def get_points(next_step, step, a, b):
    l = euclidean_distance(a, b)
    m = ((b[1] - a[1]) / (b[0] - a[0])).item()
    sign_x = -1 if b[0] < a[0] else 1
    sign_y = -1 if b[1] < a[1] else 1
    pts = []
    used = 0
    while next_step <= l:
        used += next_step
        l -= next_step
        dx = sign_x * next_step / math.sqrt(1 + m ** 2)
        dy = m * dx if abs(dx) > 1e-06 else sign_y * next_step
        a[0] += dx
        a[1] += dy
        pts.append((a[0].item(), a[1].item()))
        next_step = step
    next_step = step - l
    return (next_step, used, pts)

def euclidean_distance(a, b):
    return math.sqrt((a - b).pow(2).sum().item())

class SinkhornDistance(nn.Module):

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        import torch
        cost_matrix = self._cost_matrix(x, y)
        batch_size = 1 if x.dim() == 2 else x.shape[0]
        mu = torch.empty(batch_size, x.shape[-2], dtype=torch.float, device=x.device, requires_grad=False).fill_(1.0 / x.shape[-2]).squeeze()
        nu = torch.empty(batch_size, y.shape[-2], dtype=torch.float, device=y.device, requires_grad=False).fill_(1.0 / y.shape[-2]).squeeze()
        u, v = (torch.zeros_like(mu), torch.zeros_like(nu))
        for _ in range(self.max_iter):
            previous = u
            u = self.eps * (torch.log(mu + 1e-08) - torch.logsumexp(self.M(cost_matrix, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-08) - torch.logsumexp(self.M(cost_matrix, u, v).transpose(-2, -1), dim=-1)) + v
            error = (u - previous).abs().sum(-1).mean()
            if error.item() < 0.1:
                break
        transport = torch.exp(self.M(cost_matrix, u, v))
        cost = torch.sum(transport * cost_matrix, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return (cost.detach().reshape(1).cpu(), transport, cost_matrix)

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum(torch.abs(x_col - y_lin) ** p, -1)
        return C

def point_cloud(adjacency, nodes, n_points):
    import numpy as np
    import torch
    nodes = nodes.detach().cpu()
    adjacency = adjacency.detach().cpu()
    ordered_edges = np.argwhere(np.tril(adjacency.numpy(), k=-1) == 1.0)
    total = 0.0
    for i, j in ordered_edges:
        total += euclidean_distance(nodes[i], nodes[j])
    step = total / (n_points - 1 + 0.01)
    points = []
    next_step = 0.0
    for i, j in ordered_edges:
        next_step, used, sampled = get_points(next_step, step, nodes[j].clone(), nodes[i].clone())
        points += sampled
        last_node = nodes[i].clone()
    if 0 < len(points) < n_points:
        while len(points) < n_points:
            points.append((last_node[0].item(), last_node[1].item()))
    if not points:
        return torch.zeros((100, 2))
    return torch.FloatTensor(points)

def street_mover_distance(nodes, edges, pred_nodes, pred_edges, device, n_points=2000, eps=1e-7, max_iter=100):
    nodes, edges = (nodes.detach().cpu(), edges.detach().cpu())
    pred_nodes = pred_nodes.detach().cpu()
    if len(nodes) <= 1 or len(pred_nodes) <= 1 or np.asarray(pred_edges).size == 0:
        return 1.0
    a = torch.zeros((len(nodes), len(nodes)))
    a[edges[:, 0], edges[:, 1]] = 1
    b = torch.zeros((len(pred_nodes), len(pred_nodes)))
    b[pred_edges[:, 0], pred_edges[:, 1]] = 1
    x = point_cloud(a.T, nodes, n_points).to(device)
    y = point_cloud(b.T, pred_nodes, n_points).to(device)
    result, _, _ = SinkhornDistance(eps=eps, max_iter=max_iter, reduction='mean')(x, y)
    return result.item()
