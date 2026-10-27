import numpy as np
import math

def distance(p1, p2):
    a = p1[0] - p2[0]
    b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))
    return np.sqrt(a * a + b * b)

class RoadGraph:

    def __init__(self):
        self.nodeHash = {}
        self.nodeHashReverse = {}
        self.nodes = {}
        self.edges = {}
        self.nodeLink = {}
        self.nodeID = 0
        self.edgeID = 0
        self.edgeHash = {}
        self.edgeScore = {}
        self.nodeTerminate = {}
        self.nodeScore = {}
        self.nodeLocations = {}

    def addEdge(self, nid1, lat1, lon1, nid2, lat2, lon2, reverse=False, nodeScore1=0, nodeScore2=0, edgeScore=0):
        if nid1 not in self.nodeHash.keys():
            self.nodeHash[nid1] = self.nodeID
            self.nodeHashReverse[self.nodeID] = nid1
            self.nodes[self.nodeID] = [lat1, lon1]
            self.nodeLink[self.nodeID] = []
            self.nodeScore[self.nodeID] = nodeScore1
            self.nodeID += 1
        if nid2 not in self.nodeHash.keys():
            self.nodeHash[nid2] = self.nodeID
            self.nodeHashReverse[self.nodeID] = nid2
            self.nodes[self.nodeID] = [lat2, lon2]
            self.nodeLink[self.nodeID] = []
            self.nodeScore[self.nodeID] = nodeScore2
            self.nodeID += 1
        localid1 = self.nodeHash[nid1]
        localid2 = self.nodeHash[nid2]
        if localid1 * 10000000 + localid2 in self.edgeHash.keys():
            return
        self.edges[self.edgeID] = [localid1, localid2]
        self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
        self.edgeScore[self.edgeID] = edgeScore
        self.edgeID += 1
        if localid2 not in self.nodeLink[localid1]:
            self.nodeLink[localid1].append(localid2)
        if reverse == True:
            if localid2 not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[localid2] = []
            if localid1 not in self.nodeLinkReverse[localid2]:
                self.nodeLinkReverse[localid2].append(localid1)

    def ReverseDirectionLink(self):
        edgeList = list(self.edges.values())
        self.nodeLinkReverse = {}
        for edge in edgeList:
            localid1 = edge[1]
            localid2 = edge[0]
            if localid1 not in self.nodeLinkReverse:
                self.nodeLinkReverse[localid1] = [localid2]
            elif localid2 not in self.nodeLinkReverse[localid1]:
                self.nodeLinkReverse[localid1].append(localid2)
        for nodeId in self.nodes.keys():
            if nodeId not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[nodeId] = []

    def distanceBetweenTwoLocation(self, loc1, loc2, max_distance):
        localNodeList = {}
        localNodeDistance = {}
        localEdges = {}
        edge_covered = {}
        if loc1[0] == loc2[0] and loc1[1] == loc2[1]:
            return abs(loc1[2] - loc2[2])
        elif loc1[0] == loc2[1] and loc1[1] == loc2[0]:
            return abs(loc1[2] - loc2[3])
        ans_dist = 100000
        Queue = [(loc1[0], -1, loc1[2]), (loc1[1], -1, loc1[2])]
        while True:
            if len(Queue) == 0:
                break
            args = Queue.pop(0)
            node_cur, node_prev, dist = (args[0], args[1], args[2])
            old_node_dist = 1
            if node_cur in localNodeList.keys():
                old_node_dist = localNodeDistance[node_cur]
                if localNodeDistance[node_cur] <= dist:
                    continue
            if dist > max_distance:
                continue
            lat1 = self.nodes[node_cur][0]
            lon1 = self.nodes[node_cur][1]
            localNodeList[node_cur] = 1
            localNodeDistance[node_cur] = dist
            if node_cur not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[node_cur] = []
            reverseList = []
            reverseList = self.nodeLinkReverse[node_cur]
            visited_next_node = []
            for next_node in self.nodeLink[node_cur] + reverseList:
                if next_node == node_prev:
                    continue
                if next_node == node_cur:
                    continue
                if next_node == loc1[0] or next_node == loc1[1]:
                    continue
                if next_node in visited_next_node:
                    continue
                visited_next_node.append(next_node)
                edgeS = 0
                lat0 = 0
                lon0 = 0
                lat1 = self.nodes[node_cur][0]
                lon1 = self.nodes[node_cur][1]
                lat2 = self.nodes[next_node][0]
                lon2 = self.nodes[next_node][1]
                localEdgeId = node_cur * 10000000 + next_node
                if node_cur == loc2[0] and next_node == loc2[1]:
                    new_ans = dist + loc2[2]
                    if new_ans < ans_dist:
                        ans_dist = new_ans
                elif node_cur == loc2[1] and next_node == loc2[0]:
                    new_ans = dist + loc2[3]
                    if new_ans < ans_dist:
                        ans_dist = new_ans
                l = distance((lat2, lon2), (lat1, lon1))
                Queue.append((next_node, node_cur, dist + l))
        return ans_dist

    def TOPOWalk(self, nodeid, step=5e-05, r=0.003, direction=False, newstyle=False, nid1=0, nid2=0, dist1=0, dist2=0, bidirection=False, CheckGPS=None, metaData=None):
        localNodeList = {}
        localNodeDistance = {}
        mables = []
        localEdges = {}
        edge_covered = {}
        if newstyle == False:
            Queue = [(nodeid, -1, 0)]
        else:
            Queue = [(nid1, -1, dist1), (nid2, -1, dist2)]
        lat1 = self.nodes[nid1][0]
        lon1 = self.nodes[nid1][1]
        lat2 = self.nodes[nid2][0]
        lon2 = self.nodes[nid2][1]
        l = distance((lat2, lon2), (lat1, lon1))
        num = int(math.ceil(l / step))
        alpha = 0
        while True:
            latI = lat1 * alpha + lat2 * (1 - alpha)
            lonI = lon1 * alpha + lon2 * (1 - alpha)
            d1 = distance((latI, lonI), (lat1, lon1))
            d2 = distance((latI, lonI), (lat2, lon2))
            if dist1 - d1 < r or dist2 - d2 < r:
                if (latI, lonI, lat2 - lat1, lon2 - lon1) not in mables:
                    mables.append((latI, lonI, lat2 - lat1, lon2 - lon1))
                    if bidirection == True:
                        if nid1 in self.nodeLink[nid2] and nid2 in self.nodeLink[nid1]:
                            mables.append((latI + 1e-05, lonI + 1e-05, lat2 - lat1, lon2 - lon1))
            alpha += step / l
            if alpha > 1.0:
                break
        while True:
            if len(Queue) == 0:
                break
            args = Queue.pop(0)
            node_cur, node_prev, dist = (args[0], args[1], args[2])
            old_node_dist = 1
            if node_cur in localNodeList.keys():
                old_node_dist = localNodeDistance[node_cur]
                if localNodeDistance[node_cur] <= dist:
                    continue
            if dist > r:
                continue
            lat1 = self.nodes[node_cur][0]
            lon1 = self.nodes[node_cur][1]
            localNodeList[node_cur] = 1
            localNodeDistance[node_cur] = dist
            if node_cur not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[node_cur] = []
            reverseList = []
            if direction == False:
                reverseList = self.nodeLinkReverse[node_cur]
            visited_next_node = []
            for next_node in self.nodeLink[node_cur] + reverseList:
                if next_node == node_prev:
                    continue
                if next_node == node_cur:
                    continue
                if next_node == nid1 or next_node == nid2:
                    continue
                if next_node in visited_next_node:
                    continue
                visited_next_node.append(next_node)
                edgeS = 0
                lat0 = 0
                lon0 = 0
                lat1 = self.nodes[node_cur][0]
                lon1 = self.nodes[node_cur][1]
                lat2 = self.nodes[next_node][0]
                lon2 = self.nodes[next_node][1]
                localEdgeId = node_cur * 10000000 + next_node
                l = distance((lat2, lon2), (lat1, lon1))
                num = int(math.ceil(l / step))
                bias = step * math.ceil(dist / step) - dist
                cur = bias
                if old_node_dist + l < r:
                    Queue.append((next_node, node_cur, dist + l))
                else:
                    start_limitation = 0
                    end_limitation = l
                    if (node_cur, next_node) in edge_covered.keys():
                        start_limitation = edge_covered[node_cur, next_node]
                    if (next_node, node_cur) in edge_covered.keys():
                        end_limitation = l - edge_covered[next_node, node_cur]
                    turnnel_edge = False
                    if metaData is not None:
                        nnn1 = self.nodeHashReverse[next_node]
                        nnn2 = self.nodeHashReverse[node_cur]
                        if metaData.edgeProperty[metaData.edge2edgeid[nnn1, nnn2]]['layer'] < 0:
                            turnnel_edge = True
                    while cur < l:
                        alpha = cur / l
                        if dist + l * alpha > r:
                            break
                        if l * alpha < start_limitation:
                            cur += step
                            continue
                        if l * alpha > end_limitation:
                            break
                        latI = lat2 * alpha + lat1 * (1 - alpha)
                        lonI = lon2 * alpha + lon1 * (1 - alpha)
                        if (latI, lonI, lat2 - lat1, lon2 - lon1) not in mables and turnnel_edge is False:
                            mables.append((latI, lonI, lat2 - lat1, lon2 - lon1))
                            if bidirection == True:
                                if next_node in self.nodeLink[node_cur] and node_cur in self.nodeLink[next_node] and (turnnel_edge is False):
                                    mables.append((latI + 1e-05, lonI + 1e-05, lat2 - lat1, lon2 - lon1))
                        cur += step
                    if (node_cur, next_node) in edge_covered.keys():
                        edge_covered[node_cur, next_node] = cur - step
                    else:
                        edge_covered[node_cur, next_node] = cur - step
                    l = distance((lat2, lon2), (lat1, lon1))
                    Queue.append((next_node, node_cur, dist + l))
        result_marbles = []
        if CheckGPS is None:
            result_marbles = mables
        else:
            for mable in mables:
                if CheckGPS(mable[0], mable[1]) == True:
                    result_marbles.append(mable)
        return result_marbles
