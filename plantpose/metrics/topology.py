import numpy as np
import math
import scipy.ndimage
from . import road_graph as splfy
from rtree import index
from time import time
from hopcroftkarp import HopcroftKarp

def compute_topo(points, edges, pred_points, pred_edges, img_size=128):
    A = np.zeros((points.shape[0], points.shape[0]))
    A[edges[:, 0], edges[:, 1]] = 1
    A[edges[:, 1], edges[:, 0]] = 1
    graph = {}
    for i, n in enumerate(points):
        idx = np.nonzero(A[:, i])
        n1 = points[idx].squeeze(1)
        n1 = [tuple(n_.cpu().numpy() * img_size) for n_ in n1]
        graph.update({tuple(n.cpu().numpy() * img_size): n1})
    pred_A = np.zeros((pred_points.shape[0], pred_points.shape[0]))
    pred_A[pred_edges[:, 0], pred_edges[:, 1]] = 1
    pred_A[pred_edges[:, 1], pred_edges[:, 0]] = 1
    pred_graph = {}
    for i, n in enumerate(pred_points):
        idx = np.nonzero(pred_A[:, i])
        n1 = pred_points[idx].squeeze(1)
        n1 = [tuple(n_.cpu().numpy() * img_size) for n_ in n1]
        pred_graph.update({tuple(n.cpu().numpy() * img_size): n1})
    graph_gt = create_graph(graph)
    graph_prop = create_graph(pred_graph)
    region = [min_lat - 300 * 1.0 / 111111.0, lon_top_left - 500 * 1.0 / 111111.0, lat_top_left + 300 * 1.0 / 111111.0, max_lon + 500 * 1.0 / 111111.0]
    graph_gt.region = region
    graph_prop.region = region
    threshold = 0.0001
    losm = TOPOGenerateStartingPoints(graph_gt, region=region, image='NULL', check=False, direction=False, metaData=None)
    lmap = TOPOGeneratePairs(graph_prop, graph_gt, losm, threshold=threshold, region=region)
    r = 0.003
    if lat_top_left - min_lat < 0.01:
        r = 0.0015
    topoResult = TOPOWithPairs(graph_prop, graph_gt, lmap, losm, r=r, step=5e-05, threshold=threshold, one2oneMatching=True, metaData=None)
    return topoResult
lat_top_left = 41.0
lon_top_left = -71.0
min_lat = 41.0
max_lon = -71.0

def xy2latlon(x, y):
    lat = lat_top_left - x * 1.0 / 111111.0
    lon = lon_top_left + y * 1.0 / 111111.0 / math.cos(math.radians(lat_top_left))
    return (lat, lon)

def create_graph(m):
    global min_lat
    global max_lon
    graph = splfy.RoadGraph()
    nid = 0
    idmap = {}
    for k, v in m.items():
        n1 = k
        lat1, lon1 = xy2latlon(n1[0], n1[1])
        if lat1 < min_lat:
            min_lat = lat1
        if lon1 > max_lon:
            max_lon = lon1
        for n2 in v:
            lat2, lon2 = xy2latlon(n2[0], n2[1])
            if n1 in idmap:
                id1 = idmap[n1]
            else:
                id1 = nid
                idmap[n1] = nid
                nid = nid + 1
            if n2 in idmap:
                id2 = idmap[n2]
            else:
                id2 = nid
                idmap[n2] = nid
                nid = nid + 1
            graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)
    graph.ReverseDirectionLink()
    for node in graph.nodes.keys():
        graph.nodeScore[node] = 100
    for edge in graph.edges.keys():
        graph.edgeScore[edge] = 100
    return graph

def latlonNorm(p1, lat=40):
    p11 = p1[1] * math.cos(math.radians(lat))
    l = np.sqrt(p11 * p11 + p1[0] * p1[0])
    return (p1[0] / l, p11 / l)

def pointToLineDistance(p1, p2, p3):
    dist = np.sqrt(p2[0] * p2[0] + p2[1] * p2[1])
    proj_length = (p2[0] * p3[0] + p2[1] * p3[1]) / dist
    if proj_length > dist:
        a = p3[0] - p2[0]
        b = p3[1] - p2[1]
        return np.sqrt(a * a + b * b)
    if proj_length < 0:
        a = p3[0] - p1[0]
        b = p3[1] - p1[1]
        return np.sqrt(a * a + b * b)
    alpha = proj_length / dist
    p4 = [0, 0]
    p4[0] = alpha * p2[0]
    p4[1] = alpha * p2[1]
    a = p3[0] - p4[0]
    b = p3[1] - p4[1]
    return np.sqrt(a * a + b * b)

def pointToLineDistanceLatLon(p1, p2, p3):
    pp2 = [0, 0]
    pp3 = [0, 0]
    pp2[0] = p2[0] - p1[0]
    pp2[1] = (p2[1] - p1[1]) * math.cos(math.radians(p1[0]))
    pp3[0] = p3[0] - p1[0]
    pp3[1] = (p3[1] - p1[1]) * math.cos(math.radians(p1[0]))
    return pointToLineDistance((0, 0), pp2, pp3)

def Coord2Pixels(lat, lon, min_lat, min_lon, max_lat, max_lon, sizex, sizey):
    ilat = sizex - int((lat - min_lat) / ((max_lat - min_lat) / sizex))
    ilon = int((lon - min_lon) / ((max_lon - min_lon) / sizey))
    return (ilat, ilon)

def distance(p1, p2):
    a = p1[0] - p2[0]
    b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))
    return np.sqrt(a * a + b * b)

def TOPOGenerateStartingPoints(OSMMap, check=True, density=0.0005, region=None, image=None, direction=False, metaData=None, mergin=0.07):
    result = []
    tunnel_skip_num = 0
    svgEdges = []
    if image != 'NULL':
        img = scipy.ndimage.imread(image)
        sizex = np.shape(img)[0]
        sizey = np.shape(img)[1]
        if len(np.shape(img)) > 2:
            img = img[:, :, 3].reshape((sizex, sizey))
    else:
        img = None

    def Coord2Pixels(lat, lon, min_lat, min_lon, max_lat, max_lon, sizex, sizey):
        ilat = sizex - int((lat - min_lat) / ((max_lat - min_lat) / sizex))
        ilon = int((lon - min_lon) / ((max_lon - min_lon) / sizey))
        return (ilat, ilon)
    visitedNodes = []
    for nodeid in OSMMap.nodes.keys():
        if nodeid in visitedNodes:
            continue
        cur_node = nodeid
        next_nodes = {}
        for nn in OSMMap.nodeLink[cur_node] + OSMMap.nodeLinkReverse[cur_node]:
            next_nodes[nn] = 1
        if len(next_nodes.keys()) == 2:
            continue
        for nextnode in next_nodes.keys():
            if nextnode in visitedNodes:
                continue
            node_list = [nodeid]
            cur_node = nextnode
            while True:
                node_list.append(cur_node)
                neighbor = {}
                for nn in OSMMap.nodeLink[cur_node] + OSMMap.nodeLinkReverse[cur_node]:
                    neighbor[nn] = 1
                if len(neighbor.keys()) != 2:
                    break
                if node_list[-2] == list(neighbor.keys())[0]:
                    cur_node = list(neighbor.keys())[1]
                else:
                    cur_node = list(neighbor.keys())[0]
            for i in range(1, len(node_list) - 1):
                visitedNodes.append(node_list[i])
            dists = []
            dist = 0
            for i in range(0, len(node_list) - 1):
                dists.append(dist)
                dist += distance(OSMMap.nodes[node_list[i]], OSMMap.nodes[node_list[i + 1]])
            dists.append(dist)
            if dist < density / 2:
                continue
            n = max(int(dist / density), 1)
            alphas = [float(x + 1) / float(n + 1) for x in range(n)]
            for alpha in alphas:
                for j in range(len(node_list) - 1):
                    if metaData is not None:
                        nnn1 = OSMMap.nodeHashReverse[node_list[j]]
                        nnn2 = OSMMap.nodeHashReverse[node_list[j + 1]]
                        if metaData.edgeProperty[metaData.edge2edgeid[nnn1, nnn2]]['layer'] < 0:
                            tunnel_skip_num += 1
                            continue
                    if alpha * dist >= dists[j] and alpha * dist <= dists[j + 1]:
                        a = (alpha * dist - dists[j]) / (dists[j + 1] - dists[j])
                        lat = (1 - a) * OSMMap.nodes[node_list[j]][0] + a * OSMMap.nodes[node_list[j + 1]][0]
                        lon = (1 - a) * OSMMap.nodes[node_list[j]][1] + a * OSMMap.nodes[node_list[j + 1]][1]
                        if img != None:
                            x, y = Coord2Pixels(lat, lon, region[0], region[1], region[2], region[3], sizex, sizey)
                            if x > 0 and x < sizex and (y > 0) and (y < sizey):
                                if img[x, y] > 0:
                                    result.append((lat, lon, node_list[j], node_list[j + 1], alpha * dist - dists[j], dists[j + 1] - alpha * dist))
                        else:
                            lat_mergin = mergin * (region[2] - region[0])
                            lon_mergin = mergin * (region[3] - region[1])
                            if lat - region[0] > lat_mergin and region[2] - lat > lat_mergin and (lon - region[1] > lon_mergin) and (region[3] - lon > lon_mergin):
                                result.append((lat, lon, node_list[j], node_list[j + 1], alpha * dist - dists[j], dists[j + 1] - alpha * dist))
    for _, edge in OSMMap.edges.items():
        svgEdges.append((OSMMap.nodes[edge[0]][0], OSMMap.nodes[edge[0]][1], OSMMap.nodes[edge[1]][0], OSMMap.nodes[edge[1]][1]))
    return result

def TOPOGeneratePairs(GPSMap, OSMMap, OSMList, threshold=0.0001, region=None, single=False, edgeids=None):
    result = {}
    matchedLoc = []
    idx = index.Index()
    if edgeids is not None:
        for edgeid in edgeids:
            if edgeid not in GPSMap.edges.keys():
                continue
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]
            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]
            idx.insert(edgeid, (min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2)))
    else:
        for edgeid in GPSMap.edges.keys():
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]
            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]
            idx.insert(edgeid, (min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2)))
    for i in range(len(OSMList)):
        item = OSMList[i]
        lat = item[0]
        lon = item[1]
        possible_edges = list(idx.intersection((lat - threshold * 2, lon - threshold * 2, lat + threshold * 2, lon + threshold * 2)))
        min_dist = 10000
        min_edge = -1
        for edgeid in possible_edges:
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            n3 = item[2]
            n4 = item[3]
            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]
            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]
            lat3 = OSMMap.nodes[n3][0]
            lon3 = OSMMap.nodes[n3][1]
            lat4 = OSMMap.nodes[n4][0]
            lon4 = OSMMap.nodes[n4][1]
            nlat1, nlon1 = latlonNorm((lat2 - lat1, lon2 - lon1))
            nlat2, nlon2 = latlonNorm((lat4 - lat3, lon4 - lon3))
            dist = pointToLineDistanceLatLon((lat1, lon1), (lat2, lon2), (lat, lon))
            if dist < threshold and dist < min_dist:
                angle_dist = 1.0 - abs(nlat1 * nlat2 + nlon1 * nlon2)
                if edgeids is None:
                    if angle_dist < 0.04:
                        min_edge = edgeid
                        min_dist = dist
                else:
                    min_edge = edgeid
                    min_dist = dist
        if min_edge != -1:
            edgeid = min_edge
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]
            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]
            result[i] = [edgeid, n1, n2, distance((lat1, lon1), (lat, lon)), distance((lat2, lon2), (lat, lon)), lat, lon]
            matchedLoc.append((lat, lon))
            if single == True:
                return result
    svgEdges = []
    for _, edge in OSMMap.edges.items():
        svgEdges.append((OSMMap.nodes[edge[0]][0], OSMMap.nodes[edge[0]][1], OSMMap.nodes[edge[1]][0], OSMMap.nodes[edge[1]][1]))
    if region is not None:
        pass
    return result

def TOPO121(topo_result, roadgraph):
    rtree_index = index.Index()
    for ind in range(len(topo_result)):
        r = 1e-06
        lat = topo_result[ind][0]
        lon = topo_result[ind][1]
        rtree_index.insert(ind, [lat - r, lon - r, lat + r, lon + r])
    new_list = []
    for ind in range(len(topo_result)):
        lat = topo_result[ind][0]
        lon = topo_result[ind][1]
        r_lat = 0.0003
        r_lon = 0.0003 / math.cos(math.radians(lat))
        candidate = list(rtree_index.intersection([lat - r_lat, lon - r_lon, lat + r_lat, lon + r_lon]))
        competitors = []
        gpsn1, gpsn2, gpsd1, gpsd2 = (topo_result[ind][4], topo_result[ind][5], topo_result[ind][6], topo_result[ind][7])
        for can_id in candidate:
            t_gpsn1, t_gpsn2, t_gpsd1, t_gpsd2 = (topo_result[can_id][4], topo_result[can_id][5], topo_result[can_id][6], topo_result[can_id][7])
            d = roadgraph.distanceBetweenTwoLocation((gpsn1, gpsn2, gpsd1, gpsd2), (t_gpsn1, t_gpsn2, t_gpsd1, t_gpsd2), max_distance=0.0003)
            if d < 0.0002:
                competitors.append(can_id)
        new_list.append((topo_result[ind], ind, competitors))

    def get_key(item):
        return item[0][2]
    new_list = sorted(new_list, key=get_key)
    result = []
    mark = {}
    for ind in range(len(new_list) - 1, -1, -1):
        if new_list[ind][1] in mark:
            if new_list[ind][0][2] < 0.9:
                continue
        result.append(new_list[ind][0])
        for cc in new_list[ind][2]:
            mark[cc] = 1
    return result

def topoAvg(topo_result):
    p = 0
    r = 0
    for item in topo_result:
        p = p + item[2]
        r = r + item[3]
    if len(topo_result) == 0:
        return (0, 0)
    return (p / len(topo_result), r / len(topo_result))

def TOPOWithPairs(GPSMap, OSMMap, GPSList, OSMList, step=5e-05, r=0.003, threshold=0.00015, region=None, outputfile=None, one2oneMatching=True, metaData=None):
    i = 0
    precesion_sum = 0
    recall_sum = 0
    returnResult = []
    for k, itemGPS in GPSList.items():
        itemOSM = OSMList[k]
        gpsn1, gpsn2, gpsd1, gpsd2 = (itemGPS[1], itemGPS[2], itemGPS[3], itemGPS[4])
        osmn1, osmn2, osmd1, osmd2 = (itemOSM[2], itemOSM[3], itemOSM[4], itemOSM[5])
        osm_start_lat, osm_start_lon = (itemOSM[0], itemOSM[1])
        gps_start_lat, gps_start_lon = (itemGPS[5], itemGPS[6])
        lat = itemOSM[0]
        lon = itemOSM[1]
        ts1 = time()
        marbles = GPSMap.TOPOWalk(1, step=step, r=r, direction=False, newstyle=True, nid1=gpsn1, nid2=gpsn2, dist1=gpsd1, dist2=gpsd2)
        holes = OSMMap.TOPOWalk(1, step=step, r=r, direction=False, newstyle=True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2=osmd2, metaData=metaData)
        holes_bidirection = OSMMap.TOPOWalk(1, step=step, r=r, direction=False, newstyle=True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2=osmd2, bidirection=True, metaData=None)
        ts2 = time()
        idx_marbles = index.Index()
        idx_holes = index.Index()
        idx_holes_bidirection = index.Index()
        for j in range(len(marbles)):
            idx_marbles.insert(j, (marbles[j][0] - 1e-05, marbles[j][1] - 1e-05, marbles[j][0] + 1e-05, marbles[j][1] + 1e-05))
        for j in range(len(holes)):
            idx_holes.insert(j, (holes[j][0] - 1e-05, holes[j][1] - 1e-05, holes[j][0] + 1e-05, holes[j][1] + 1e-05))
        for j in range(len(holes_bidirection)):
            idx_holes_bidirection.insert(j, (holes_bidirection[j][0] - 1e-05, holes_bidirection[j][1] - 1e-05, holes_bidirection[j][0] + 1e-05, holes_bidirection[j][1] + 1e-05))
        matchedNum = 0
        bigraph = {}
        matched_marbles = []
        bipartite_graph = []
        cost_map = {}
        for marble in marbles:
            rr = threshold * 1.8
            possible_holes = list(idx_holes_bidirection.intersection((marble[0] - rr, marble[1] - rr, marble[0] + rr, marble[1] + rr)))
            for hole_id in possible_holes:
                hole = holes_bidirection[hole_id]
                ddd = distance(marble, hole)
                n1 = latlonNorm((marble[2], marble[3]))
                n2 = latlonNorm((hole[2], hole[3]))
                if marble[2] != marble[3] and hole[2] != hole[3]:
                    angle_d = 1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])
                else:
                    angle_d = 0.0
                if ddd < threshold and angle_d < 0.29:
                    if marble in bigraph.keys():
                        bigraph[marble].add(hole_id)
                    else:
                        bigraph[marble] = set([hole_id])
                    bipartite_graph.append((marble, hole_id, ddd))
                    matchedNum += 1
                    matched_marbles.append(marble)
        soft_matchedNum = 0
        if one2oneMatching == True:
            matches = HopcroftKarp(bigraph).maximum_matching()
            matchedNum = len(matches.keys()) / 2
        if len(marbles) == 0 or len(holes) == 0:
            continue
        precesion = float(matchedNum) / len(marbles)
        matchedNum = 0
        bigraph = {}
        cost_map = {}
        for hole in holes:
            rr = threshold * 1.8
            possible_marbles = list(idx_marbles.intersection((hole[0] - rr, hole[1] - rr, hole[0] + rr, hole[1] + rr)))
            for marble_id in possible_marbles:
                marble = marbles[marble_id]
                ddd = distance(marble, hole)
                n1 = latlonNorm((marble[2], marble[3]))
                n2 = latlonNorm((hole[2], hole[3]))
                if marble[2] != marble[3] and hole[2] != hole[3]:
                    angle_d = 1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])
                else:
                    angle_d = 0.0
                if ddd < threshold and angle_d < 0.29:
                    if hole in bigraph.keys():
                        bigraph[hole].add(marble_id)
                    else:
                        bigraph[hole] = set([marble_id])
                    matchedNum += 1
        soft_matchedNum = 0
        if one2oneMatching == True:
            matches = HopcroftKarp(bigraph).maximum_matching()
            matchedNum = len(matches.keys()) / 2
        recall = float(matchedNum) / len(holes)
        precesion_sum += precesion
        recall_sum += recall
        ts3 = time()
        returnResult.append((lat, lon, precesion, recall, gpsn1, gpsn2, gpsd1, gpsd2))
        i = i + 1
    new_topoResult = TOPO121(returnResult, GPSMap)
    p, r = topoAvg(new_topoResult)
    return (p, r)
