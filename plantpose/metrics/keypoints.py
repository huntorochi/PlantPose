import torch
import numpy as np
import networkx as nx

def keypoint_graph(nodes, edges, pred_nodes, pred_edges, pred_nodes_box, pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class, synthetic=False):
    out_nodes, out_edges, out_pred_nodes, out_pred_edges, out_pred_nodes_box = ([], [], [], [], [])
    out_pred_nodes_box_score, out_pred_nodes_box_class, out_pred_edges_box_score = ([], [], [])
    out_pred_edges_box_class = []
    for batch_num in range(len(nodes)):
        test_pred_graph = nx.Graph()
        for i, no in enumerate(pred_nodes[batch_num]):
            no_num = i
            test_pred_graph.add_node(no_num, node_box=pred_nodes_box[batch_num][i], node_score=pred_nodes_box_score[batch_num][i], node_class=pred_nodes_box_class[batch_num][i], node=no.cpu().detach().numpy())
        for i, ed in enumerate(pred_edges[batch_num]):
            test_pred_graph.add_edge(ed[0], ed[1], edge_score=pred_edges_box_score[batch_num][i], edge_class=pred_edges_box_class[batch_num][i])
        pred_degree_list = []
        for i in range(len(pred_nodes[batch_num])):
            if test_pred_graph.degree(i) == 2:
                pred_degree_list.append(i)
        for skip_node in pred_degree_list:
            if test_pred_graph.degree(skip_node) == 2:
                edge_data_list = list(test_pred_graph.edges(skip_node, data=True))
                new_edge_score = (edge_data_list[0][2]['edge_score'] + edge_data_list[1][2]['edge_score']) / 2
                new_edge_class = (edge_data_list[0][2]['edge_class'] + edge_data_list[1][2]['edge_class'] + 0.5) / 2
                neighbors_list = [n for n in test_pred_graph.neighbors(skip_node)]
                test_pred_graph.remove_node(skip_node)
                test_pred_graph.add_edge(neighbors_list[0], neighbors_list[1], edge_score=np.float32(new_edge_score), edge_class=np.int32(new_edge_class))
        networkx_pred_nodes = list(test_pred_graph.nodes(data=True))
        networkx_pred_edges = list(test_pred_graph.edges(data=True))
        nodes_dict = {}
        for i, data in enumerate(networkx_pred_nodes):
            nodes_dict[data[0]] = i
        batch_pred_nodes_tensor = torch.tensor([i[1]['node'] for i in networkx_pred_nodes]).to(pred_nodes[batch_num].device)
        if batch_pred_nodes_tensor.dim() != 0 and batch_pred_nodes_tensor.nelement() != 0 and (batch_pred_nodes_tensor.shape[0] > 1) and ((len(networkx_pred_edges) if not synthetic else 1) > 0):
            out_pred_nodes.append(batch_pred_nodes_tensor)
            batch_pred_edges_array = np.array([[nodes_dict[i[0]], nodes_dict[i[1]]] for i in networkx_pred_edges], dtype=np.int64)
            out_pred_edges.append(batch_pred_edges_array)
            batch_pred_nodes_box_array = np.array([i[1]['node_box'] for i in networkx_pred_nodes])
            out_pred_nodes_box.append(batch_pred_nodes_box_array)
            batch_pred_nodes_box_score_array = np.array([i[1]['node_score'] for i in networkx_pred_nodes])
            out_pred_nodes_box_score.append(batch_pred_nodes_box_score_array)
            batch_pred_nodes_box_class_array = np.array([i[1]['node_class'] for i in networkx_pred_nodes])
            out_pred_nodes_box_class.append(batch_pred_nodes_box_class_array)
            batch_pred_edges_box_score_array = np.array([i[2]['edge_score'] for i in networkx_pred_edges])
            out_pred_edges_box_score.append(batch_pred_edges_box_score_array)
            batch_pred_edges_box_class_array = np.array([i[2]['edge_class'] for i in networkx_pred_edges], dtype=np.int64)
            out_pred_edges_box_class.append(batch_pred_edges_box_class_array)
        else:
            out_pred_nodes.append(torch.empty(0, 2).to(pred_nodes[batch_num].device))
            out_pred_edges.append(torch.empty(0, 2).cpu())
            out_pred_nodes_box.append(torch.empty(0, 4).cpu().numpy())
            out_pred_nodes_box_score.append(torch.empty(0, 1).cpu().numpy())
            out_pred_nodes_box_class.append(torch.empty(0, 1).cpu().numpy())
            out_pred_edges_box_score.append(torch.empty(0, 1).cpu().numpy())
            out_pred_edges_box_class.append(torch.empty(0, 1).cpu().numpy())
        test_real_graph = nx.Graph()
        for i, no in enumerate(nodes[batch_num]):
            no_num = i
            test_real_graph.add_node(no_num, node=no.cpu().detach().numpy())
        for i, ed in enumerate(edges[batch_num]):
            a = int(ed[0].cpu().detach().numpy())
            b = int(ed[1].cpu().detach().numpy())
            test_real_graph.add_edge(a, b)
        real_degree_list = []
        for i in range(len(nodes[batch_num])):
            if test_real_graph.degree(i) == 2:
                real_degree_list.append(i)
        for skip_node in real_degree_list:
            neighbors_list = [n for n in test_real_graph.neighbors(skip_node)]
            test_real_graph.remove_node(skip_node)
            test_real_graph.add_edge(neighbors_list[0], neighbors_list[1])
        networkx_real_nodes = list(test_real_graph.nodes(data=True))
        networkx_real_edges = list(test_real_graph.edges(data=True))
        nodes_dict = {}
        for i, data in enumerate(networkx_real_nodes):
            nodes_dict[data[0]] = i
        batch_real_nodes_tensor = torch.tensor([i[1]['node'] for i in networkx_real_nodes]).to(nodes[batch_num].device)
        out_nodes.append(batch_real_nodes_tensor)
        batch_real_edges_tensor = torch.tensor([[nodes_dict[i[0]], nodes_dict[i[1]]] for i in networkx_real_edges]).to(edges[batch_num].device)
        out_edges.append(batch_real_edges_tensor)
    return (out_nodes, out_edges, out_pred_nodes, out_pred_edges, out_pred_nodes_box, out_pred_nodes_box_score, out_pred_nodes_box_class, out_pred_edges_box_score, out_pred_edges_box_class)
