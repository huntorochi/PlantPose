"""Image and graph augmentation for Guyot and mix."""
import copy
import random
import cv2
import networkx as nx
import numpy as np
import torch
from skimage import exposure

class GraphAugmentation:

    def _gasuss_noise(self, image, mu=0.0, sigma=0.1):
        gasuss_img = copy.deepcopy(image)
        gasuss_img = gasuss_img.astype(np.float32)
        noise = np.random.normal(mu, sigma, gasuss_img.shape)
        gauss_noise = gasuss_img + noise
        gauss_noise = np.clip(gauss_noise, 0.0, 1.0)
        return gauss_noise

    def _addNoise(self, img):
        noise_img = copy.deepcopy(img)
        return self._gasuss_noise(noise_img)

    def _flip2(self, img, nodes_list):
        flip_nodes_list = copy.deepcopy(nodes_list)
        flip_img = copy.deepcopy(img)
        w = flip_img.shape[1]
        img2 = cv2.flip(flip_img, 1)
        flip_new_nodes_list = list()
        for x, y in flip_nodes_list:
            flip_new_nodes_list.append([w - x, y])
        return (img2, flip_new_nodes_list)

    def _changeLight(self, img):
        flag = random.uniform(1 - self.max_change_light_rate, 1 + self.max_change_light_rate)
        light_img = copy.deepcopy(img)
        return exposure.adjust_gamma(light_img, flag)

    def _augment_one_sample(self, check_img, nodes_list):
        height, width, channels = check_img.shape
        a = random.random()
        if a < 0.2:
            crop_img = self._changeLight(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)
        elif 0.2 <= a < 0.3:
            crop_img = self._addNoise(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)
        else:
            c = random.random()
            if c < 0.8:
                crop_img = self._changeLight(check_img)
                crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
            elif 0.8 <= c < 0.9:
                crop_img = self._addNoise(check_img)
                crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
            else:
                crop_img = self._changeLight(check_img)
                crop_img = self._addNoise(crop_img)
                crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
        output_nodes = np.array(nodes_list_check)
        if crop_img.shape[0] == height and crop_img.shape[1] == width:
            output_nodes = output_nodes / np.array([width, height])
            return [1, crop_img, output_nodes, 0]
        else:
            new_height, new_width = (crop_img.shape[0], crop_img.shape[1])
            output_nodes = output_nodes / np.array([new_width, new_height])
            new_img = cv2.resize(crop_img, (width, height))
            return [1, new_img, output_nodes, 0]

    def _add_edge_branch(self, img, nodes_tensor, connect_tensor, final_nodes_tensor, final_connect_tensor, M):

        def get_children(graph, node):
            out_edges = graph.out_edges(node)
            children = [target for source, target in out_edges]
            return children

        def find_node_by_point(graph, point):
            for node, data in graph.nodes(data=True):
                if data['point'] == point:
                    return node
            return None
        old_img = copy.deepcopy(img).cpu().numpy()
        C, height, width = old_img.shape
        edge_func = {'top': [0, 1, 0], 'bottom': [0, 1, -height], 'left': [1, 0, 0], 'right': [1, 0, -width]}
        check_nodes_list = list()
        for x, y in (nodes_tensor * torch.tensor([width, height], device=nodes_tensor.device)).cpu().numpy():
            x1 = M[0][0] * x + M[0][1] * y + M[0][2]
            y1 = M[1][0] * x + M[1][1] * y + M[1][2]
            check_nodes_list.append([int(x1), int(y1)])
        check_G = nx.DiGraph()
        rotate_G = nx.DiGraph()
        for i, point in enumerate(check_nodes_list):
            check_G.add_node(i, point=point)
        for connection in connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in check_G.nodes and end in check_G.nodes:
                check_G.add_edge(start, end)
        rotate_nodes_list = [[int(x * width), int(y * height)] for x, y in final_nodes_tensor.cpu().numpy()]
        for i, point in enumerate(rotate_nodes_list):
            rotate_G.add_node(i, point=point)
        for connection in final_connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in rotate_G.nodes and end in rotate_G.nodes:
                rotate_G.add_edge(start, end)
        check_nodes_data = [check_G.nodes[node]['point'] for node in check_G.nodes]
        rotate_nodes_data = [rotate_G.nodes[node]['point'] for node in rotate_G.nodes]
        check_nodes_array = np.array(check_nodes_data)
        rotate_nodes_array = np.array(rotate_nodes_data)
        updated_check_nodes_data = check_nodes_data.copy()
        for rotate_node in rotate_nodes_array:
            distances = np.linalg.norm(check_nodes_array - rotate_node, axis=1)
            nearest_idx = np.argmin(distances)
            updated_check_nodes_data[nearest_idx] = list(rotate_node)
        check_G.clear()
        for i, point in enumerate(updated_check_nodes_data):
            check_G.add_node(i, point=point)
        for connection in connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in check_G.nodes and end in check_G.nodes:
                check_G.add_edge(start, end)
        check_nodes_data = [check_G.nodes[node]['point'] for node in check_G.nodes]
        rotate_start_node = 0
        rotate_end_nodes = [node for node, degree in rotate_G.degree() if degree == 1 and node != rotate_start_node]
        for rotate_end_node in rotate_end_nodes:
            rotate_end_point = rotate_G.nodes[rotate_end_node]['point']
            check_node = find_node_by_point(check_G, rotate_end_point)
            if check_node is not None:
                check_children = get_children(check_G, check_node)
                if len(check_children) == 1:
                    check_children_point_x, check_children_point_y = check_G.nodes[check_children[0]]['point']
                    distance_to_edges = {'top': abs(check_children_point_y), 'bottom': abs(height - check_children_point_y), 'left': abs(check_children_point_x), 'right': abs(width - check_children_point_x)}
                    closest_edge_key, shortest_distance = min(distance_to_edges.items(), key=lambda item: item[1])
                    closest_edge_func = edge_func[closest_edge_key]
                    x1, y1 = check_G.nodes[check_children[0]]['point']
                    x2, y2 = rotate_end_point
                    A, B, C = closest_edge_func
                    denominator = A * (x2 - x1) + B * (y2 - y1)
                    if denominator != 0:
                        t = -(A * x1 + B * y1 + C) / denominator
                        intersection_x = x1 + t * (x2 - x1)
                        intersection_y = y1 + t * (y2 - y1)
                        clamped_intersection_x = np.clip(int(intersection_x), 0, width)
                        clamped_intersection_y = np.clip(int(intersection_y), 0, height)
                        new_child_point = [int(clamped_intersection_x), int(clamped_intersection_y)]
                        distance = ((new_child_point[0] - rotate_end_point[0]) ** 2 + (new_child_point[1] - rotate_end_point[1]) ** 2) ** 0.5
                        if new_child_point not in check_nodes_data and 3 < distance < 16:
                            rotate_G.add_node(len(rotate_G.nodes), point=new_child_point)
                            rotate_G.add_edge(rotate_end_node, len(rotate_G.nodes) - 1)
        rotate_nodes_data = [rotate_G.nodes[node]['point'] for node in rotate_G.nodes]
        final_rotate_nodes_tensor = torch.tensor(rotate_nodes_data, dtype=torch.float32, device=nodes_tensor.device) / torch.tensor([width, height], dtype=torch.float32, device=nodes_tensor.device)
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(rotate_G.nodes)}
        updated_edges = [(index_mapping[start], index_mapping[end]) for start, end in rotate_G.edges]
        final_rotate_connect_tensor = torch.tensor(updated_edges, dtype=torch.long, device=connect_tensor.device)
        return (final_rotate_nodes_tensor, final_rotate_connect_tensor)

    def _rotate(self, img, nodes_tensor, connect_tensor):
        rotate_nodes_tensor = copy.deepcopy(nodes_tensor)
        rotate_img = copy.deepcopy(img).cpu().numpy()
        C, height, width = rotate_img.shape
        rotate_img = np.transpose(rotate_img, (1, 2, 0))
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        img2 = cv2.warpAffine(rotate_img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rotate_new_nodes_list = list()
        for x, y in (rotate_nodes_tensor * torch.tensor([width, height], device=rotate_nodes_tensor.device)).cpu().numpy():
            x1 = M[0][0] * x + M[0][1] * y + M[0][2]
            y1 = M[1][0] * x + M[1][1] * y + M[1][2]
            rotate_new_nodes_list.append([int(x1), int(y1)])
        G = nx.Graph()
        for i, point in enumerate(rotate_new_nodes_list):
            G.add_node(i, point=point)
        for connection in connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in G.nodes and end in G.nodes:
                G.add_edge(start, end)
        for node in rotate_new_nodes_list:
            x, y = node
            if not (0 <= x < width and 0 <= y < height):
                index_id = rotate_new_nodes_list.index(node)
                if index_id in G.nodes:
                    G.remove_node(index_id)
        nodes_data = [G.nodes[node]['point'] for node in G.nodes]
        final_nodes_tensor = torch.tensor(nodes_data, dtype=torch.float32, device=nodes_tensor.device) / torch.tensor([width, height], dtype=torch.float32, device=nodes_tensor.device)
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(G.nodes)}
        updated_edges = [(index_mapping[start], index_mapping[end]) for start, end in G.edges]
        final_connect_tensor = torch.tensor(updated_edges, dtype=torch.long, device=connect_tensor.device)
        img2 = np.transpose(img2, (2, 0, 1))
        img2_tensor = torch.tensor(img2, dtype=torch.float32, device=nodes_tensor.device)
        final_nodes_tensor, final_connect_tensor = self._add_edge_branch(img, nodes_tensor, connect_tensor, final_nodes_tensor, final_connect_tensor, M)
        return (img2_tensor, final_nodes_tensor, final_connect_tensor, M)
