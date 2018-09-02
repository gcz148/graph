import math
from collections import deque
import heapq
import pprint
import re

class Graph_node:
    def __init__(self, name, data, extra=None):
        self.name = name
        self.data = data
        self.extra = extra
        self.adj_node = []
        self.start_edge = []
        self.end_edge = []
        self.prev_node = []

    def add_adj(self, node):
        self.adj_node.append(node)

    def add_start_edge(self, edge):
        self.start_edge.append(edge)

    def add_end_edge(self, edge):
        self.end_edge.append(edge)

    def add_prev(self, prev):
        self.prev_node.append(prev)

class Graph_edge:
    def __init__(self, node1, node2, max_flow=math.inf, cur_flow=math.inf):
        self.start = node1
        self.end = node2
        self.max_flow = max_flow
        self.cur_flow = cur_flow

class Graph:
    graph_count = 0
    def __init__(self, graph_data, is_direct=False, data_type="graph"):
        Graph.graph_count += 1
        self.name = "graph_"+str(Graph.graph_count)
        self.is_direct = is_direct
        if data_type == "graph":
            self.init_from_graph(graph_data)
        elif data_type == "matrix":
            self.init_from_matrix(graph_data)
        elif data_type == "list":
            self.init_from_list(graph_data)
        elif data_type == "file":
            self.init_from_file(graph_data)
        else:
            self.nodes = []
            self.edges = []

    def init_from_graph(self, graph_data):
        self.edges = graph_data.edges[:]
        self.nodes = graph_data.nodes[:]
        self.hash_map = graph_data.hash_map.copy()
        self.edge_map = graph_data.edge_map.copy()
        self.is_direct = graph_data.is_direct

    def init_from_matrix(self, graph_data):
        graph_data_length = len(graph_data)
        self.edges = []
        self.nodes = [Graph_node(i, i) for i in range(graph_data_length)]
        self.hash_map = {i: i for i in range(graph_data_length)}
        self.edge_map = {}
        for i in range(graph_data_length):
            for j in range(graph_data_length):
                if not math.isinf(graph_data[i][j]):
                    self.nodes[i].add_adj(self.nodes[j])
                    self.edges.append(Graph_edge(self.nodes[i], self.nodes[j], graph_data[i][j], 0))
                    self.nodes[i].add_start_edge(self.edges[-1])
                    self.nodes[j].add_end_edge(self.edges[-1])
                    self.nodes[j].add_prev(self.nodes[i])
                    self.edge_map[str(i) + "+" + str(j)] = self.edges[-1]
                    
    def init_from_list(self, graph_data):
        graph_data_length = len(graph_data)
        self.edges = []
        self.nodes = []
        self.hash_map = {}
        self.edge_map = {}
        for i in range(graph_data_length):
            self.nodes.append(Graph_node(i, graph_data[i][0]))
            self.hash_map[graph_data[i][0]] = i
        for i in range(graph_data_length):
            if len(graph_data[i]) == 2:
                for value, weight in graph_data[i][1]:
                    #print(value, weight)
                    self.nodes[i].add_adj(self.nodes[self.hash_map[value]])
                    self.edges.append(Graph_edge(self.nodes[i], self.nodes[self.hash_map[value]], weight, 0))
                    self.nodes[i].add_start_edge(self.edges[-1])
                    self.nodes[self.hash_map[value]].add_end_edge(self.edges[-1])
                    self.nodes[self.hash_map[value]].add_prev(self.nodes[i])
                    self.edge_map[str(i) + "+" + str(self.hash_map[value])] = self.edges[-1]

    def init_from_file(self, file_name):
        f = open(file_name, "r+")
        try:
            lines = f.readline()
        except UnicodeDecodeError:
            f.close()
            f = open(file_name, "r+", encoding="utf-8")
        string = f.read()
        f.close()
        v_name = []
        s_step = []
        s_length = len(string)
        t_1 = re.split("(#start)|(#vn)|(#vs)|(#vp)|(#v)|(#end)|(#solve)|(#e)", string, re.MULTILINE)
        t_2 = []
        state = ""
        start = 0
        steps = 1
        
        for value in t_1:
            if value and value != "\n":
                t_2.append(value)
        #print(t_2)
        for step in t_2:
            step_2 = step.strip("\r\n\t ")
            if step_2 == "#v":
                state = "v"
            elif step_2 == "#vn":
                state = "vn"
            elif step_2 == "#vs":
                state = "vs"
            elif step_2 == "#vp":
                state = "vp"
            elif step_2 == "#e":
                state = "e"
            elif step_2 == "#solve":
                state = "solve"
            elif state == "v":
                self.nodes = [Graph_node(i, i) for i in range(int(step_2))]
                self.hash_map = {str(i): i for i in range(len(self.nodes))}
            elif state == "vn":
                v_name = re.split("[ \r\n\t,]*", step_2, re.MULTILINE)
                for index, name in enumerate(v_name):
                    if index < len(self.nodes):
                        self.nodes[index].extra = value
            elif state == "vs":
                if not step_2:
                    continue
                temp = re.split("[ \r\n\t,]*", step_2, re.MULTILINE)
                if temp[0]:
                    start, steps = temp
                    start = int(start)
                    steps = int(step)
            elif state == "vp":
                for index, node in enumerate(self.nodes):
                    node.data = start
                    self.hash_map[step_2 + str(start)] = index
                    start += steps
            elif state == "e":
                self.edges = []
                self.edge_map = {}
                print(step_2)
                e_lines = re.split("[\r\n,]*", step_2)
                for value in e_lines:
                    if not value:
                        continue
                    #print(value)
                    temp = re.split("[ \t]*", value)
                    print(temp)
                    if len(temp) == 2:
                        s1, e1 = temp
                        weight = 1
                    elif len(temp) == 3:
                        s1, e1, weight = temp
                    elif len(temp) == 0 or len(temp) == 1:
                        #print(value)
                        continue
                    print(s1, e1)
                    n1 = self.hash_map[s1]
                    n2 = self.hash_map[e1]
                    weight = int(weight)
                    self.edges.append(Graph_edge(self.nodes[n1], self.nodes[n2], weight, 0))
                    self.nodes[n1].add_adj(self.nodes[n2])
                    self.nodes[n1].add_start_edge(self.edges[-1])
                    self.nodes[n2].add_end_edge(self.edges[-1])
                    self.nodes[n2].add_prev(self.nodes[n1])
                    self.edge_map[str(n1) + "+" + str(n2)] = self.edges[-1]
            elif state == "solve":
                s_step = re.split("[\r\n,]*", step_2, re.MULTILINE)
                state = "end"
        for step in s_step:
            step_2 = step.strip("\r\n\t ")
            if step_2 == "print":
                print(self.__str__())
            elif step_2 == "dfs":
                self.dfs()
            elif step_2 == "bfs":
                self.bfs()
            elif step_2.startswith("max"):
                _, start, end = re.split("[ \t,]*", step_2, re.MULTILINE)
                start = self.hash_map[start]
                end = self.hash_map[end]
                print(start, end)
                self.max_net_flow(self.nodes[int(start)], self.nodes[int(end)])
  
    def output_to_matrix(self):
        ret = [[math.inf for j in range(len(self.nodes))] for i in range(len(self.nodes))]
        for edge in self.edges:
            i = edge.start.name
            j = edge.end.name
            ret[i][j] = edge.weight

        return ret

    def output_to_list(self):
        ret = [[node.name] for node in self.nodes]
        for edge in self.edges:
            i = edge.start.name
            j = edge.end.name
            if len(ret[i]) == 1:
                ret[i].append([])
            ret[i][1].append([j, edge.weight])
        return ret

    def add_node(self, data):
        if isinstance(data, Graph_node):
            if data.data not in self.hash_map:
                self.hash_map[data.data] = len(self.nodes)
                self.nodes.append(Graph_node(len(self.nodes), data.data))
                return self.nodes[-1]
            else:
                return self.nodes[self.hash_map[data.data]]
        else:
            if data not in self.hash_map:
                self.hash_map[data] = len(self.nodes)
                self.nodes.append(Graph_node(len(self.nodes), data))
                return self.nodes[-1]
            else:
                return self.nodes[self.hash_map[data]]
    
    def add_edge(self, node1, node2, weight):
        node1 = self.add_node(node1)
        node2 = self.add_node(node2)
        e12 = str(node1.name) + "+" + str(node2.name)
        if e12 not in self.edge_map:
            self.edges.append(Graph_edge(node1, node2, weight, 0))
            node1.add_start_edge(self.edges[-1])
            node2.add_end_edge(self.edges[-1])
            node1.add_adj(node2)
            node2.add_prev(node1)
            return self.edges[-1]
        else:
            return self.hash_map[e12]

    def find_node(self, node):
        if isinstance(node, Graph_node):
            if node.data not in self.hash_map:
                return None
            else:
                return self.nodes[self.hash_map[node.data]]
        else:
            if node not in self.hash_map:
                return None
            else:
                return self.nodes[self.hash_map[node]]

    def find_edge(self, node1, node2):
        n1 = self.find_node(node1)
        if not n1:
            return None
        n2 = self.find_node(node_2)
        if not n2:
            return None
        e12 = str(n1.name) + "+" + str(n2.name)
        if e12 not in self.edge_map:
            return None
        else:
            return self.edge_map[e12]

    def __str__(self):
        ret = []
        ret.append("the name of graph: {0}".format(self.name))
        ret.append("the number of graph node: {0}".format(len(self.nodes)))
        for node in self.nodes:
            ret.append("the basic node {0} and its adj_node".format(node.name))
            ret2 = []
            for a_node in node.adj_node:
                ret2.append(str(a_node.name))
            ret.append(",".join(ret2))
        ret.append("the number of graph edge: {0}".format(len(self.edges)))
        for count, edge in enumerate(self.edges):
            node1 = edge.start
            node2 = edge.end
            weight = edge.max_flow
            if self.is_direct:
                ret.append("the {0} graph edge: the start node is {1}, the end node is {2}, the weight is {3}".format(
                count, node1.data, node2.data, weight))
            else:
                ret.append("the {0} graph edge: the node1 is {1}, the node2 is {2}, the weight is {3}".format(
                count, node1.data, node2.data, weight))
        ret.append("---graph end---")
        return "\n".join(ret)

    def dfs(self):
        if self.is_empty():
            return
        visited = [False] * len(self.nodes)

        def _dfs(node):
            print("the current node is {0}".format(node.name))
            visited[node.name] = True
            for a_node in node.adj_node:
                if not visited[a_node.name]:
                    _dfs(a_node)

        for node in self.nodes:
            if not visited[node.name]:
                _dfs(node)

    def is_empty(self):
        return not self.nodes

    def bfs(self):
        if self.is_empty():
            return
        
        visited = [-1] * len(self.nodes)

        def _bfs(root):
            Q = deque()
            Q.append(root)
            while Q:
                node = Q.popleft()
                print("the current node is {0}".format(node.name))
                visited[node.name] = 1
                for a_node in node.adj_node:
                    if visited[a_node.name] == -1:
                        Q.append(a_node)
                        visited[a_node.name] = 0

        for node in self.nodes:
            if visited[node.name] == -1:
                _bfs(node)

    def max_net_flow(self, start_node, end_node):

        def _bfs(start_node, end_node):
            Q = deque()
            Q.append(start_node)
            while Q and labeled[end_node.name][0] == -1:
                node = Q.popleft()
                for edge in node.start_edge:
                    temp_node = edge.end
                    temp_label = labeled[temp_node.name]
                    if temp_label[0] == -1:
                        if edge.cur_flow < edge.max_flow:
                            delta_y = min(labeled[node.name][2], edge.max_flow - edge.cur_flow)
                            temp_label[0] = 0
                            temp_label[1] = node.name
                            temp_label[2] = delta_y
                            Q.append(temp_node)
                for edge in node.end_edge:
                    temp_node = edge.start
                    temp_label = labeled[temp_node.name]
                    if temp_label[0] == -1:
                        if edge.cur_flow > 0:
                            delta_y = min(labeled[node.name][2], edge.cur_flow)
                            temp_label[0] = 0
                            temp_label[1] = -node.name
                            temp_label[2] = delta_y
                            Q.append(temp_node)
                labeled[node.name][0] = 1

        while True:
            labeled = [[-1, -1, math.inf] for i in range(len(self.nodes))]
            labeled[start_node.name][0] = 0
            _bfs(start_node, end_node)
            if labeled[end_node.name][0] == -1 or labeled[end_node.name][2] == 0:
                break
            node_1 = end_node.name
            node_prev = abs(labeled[node_1][1])
            alpha = labeled[node_1][2]
            while True:
                e12 = str(node_prev) + "+" + str(node_1)
                e21 = str(node_1) + "+" + str(node_prev)
                if e12 in self.edge_map:
                    if not math.isinf(self.edge_map[e12].cur_flow):
                        self.edge_map[e12].cur_flow += alpha
                elif e21 in self.edge_map:
                    self.edge_map[e21].cur_flow -= alpha
                if node_prev == 0:
                    break
                node_1 = node_prev
                node_prev = abs(labeled[node_prev][1])
            print(labeled)
        max_flow = 0
        print("---the max flow solve---")
        for edge in self.edges:
            if edge.start.name == start_node.name and not math.isinf(edge.cur_flow):
                max_flow += edge.cur_flow
            if not math.isinf(edge.cur_flow):
                print("the flow {0} -> {1} is: {2}".format(edge.start.name, edge.end.name, edge.cur_flow))
        print("the max flow is: {0}".format(max_flow))
        print("---the end---")

def test():
    data = [[0, [[1,8],[2,4]]], [1, [[3,2],[4,2]]], [2, [[1,4],[3,1],[4,4]]], [3, [[4,6],[5,9]]], [4, [[5,7]]], [5, []]
            ]
    graph_1 = Graph(data, data_type="list")
    print(graph_1)
    graph_1.max_net_flow(graph_1.nodes[0], graph_1.nodes[-1])
    graph_2 = Graph("C:\\Users\\Administrator\\Desktop\\aaa.txt", data_type="file")
    

if __name__ == "__main__":
    test()
