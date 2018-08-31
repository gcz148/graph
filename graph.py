import math
from collections import deque
import heapq
import pprint


class Graph_node:
    def __init__(self, name, data, extra=None):
        self.name = name
        self.data = data
        self.extra = extra
        self.adj_node = []

    def add_adj(self, node):
        self.adj_node.append(node)

class Graph_edge:
    def __init__(self, node1, node2, weight=1):
        self.start = node1
        self.end = node2
        self.weight = weight


"""
reference to the page:
https://github.com/CyC2018/Interview-Notebook/blob/master/notes/%E7%AE%97%E6%B3%95.md#%E5%9B%9B%E5%B9%B6%E6%9F%A5%E9%9B%86
"""
class UnionSet:
    def __init__(self, total):
        """construct the unionset of total node
"""
        self.id = [i for i in range(total)]
        #self.extra=[None for _ in range(total)]

    def connected(self, p, q):
        """decide the connection of node p and node q
"""
        return self.find(p) == self.find(q)

    def find(self, p):
       """find the connection part of node p
"""
       pass

    def union(self, p, q):
        """connect the node p and the node q
"""
        pass

class QuickFindUnionSet(UnionSet):
    def __init__(self, total):
        super(QuickFindUnionSet, self).__init__(total)

    def find(self, p):
        return id[p]

    def union(self, p, q):
        pID = self.find(p)
        qID = self.find(q)
        if pID == qID:
            return

        for i in range(len(self.id)):
            if self.id[i] == pID:
                self.id[i] = qID


class QuickUnionUnionSet(UnionSet):
    def __init__(self, total):
        super(QuickUnionUnionSet, self).__init__(total)

    def find(self, p):
        while p != self.find[p]:
            p = self.id[p]

        return p

    def union(self, p, q):
        pID = self.find(p)
        qID = self.find(q)
        if pID != qID:
            self.id[pID] = qID


class WeightQuickUnionSet(UnionSet):
    def __init__(self, total):
        super(WeightQuickUnionSet, self).__init__(total)
        self.weight = [1 for _ in range(total)]

    def find(self, p):
        while p != self.find[p]:
            p = self.id[p]

        return p

    def union(self, p, q):
        pID = self.find(p)
        qID = self.find(q)
        if pID == qID:
            return
        if self.weight[pID] < self.weight[qID]:
            self.id[pID] = qID
            self.weight[qID] += self.weight[pID]
        else:
            self.id[qID] = pID
            self.weight[pID] += self.weight[qID]

class WeightPathQuickUnionSet(UnionSet):
    def __init__(self, total):
        super(WeightPathQuickUnionSet, self).__init__(total)
        self.weight = [1 for _ in range(total)]

    def find(self, p):
        r = p
        while r != self.id[r]: #find the root node of tree about r
            r = self.id[r]

        k = p
        while k != r: # set the root node of parent node of node k to r
            j = self.id[k]
            self.id[k] = r
            k = j

        return r       

    def union(self, p, q):
        pID = self.find(p)
        qID = self.find(q)
        if pID == qID:
            return
        if self.weight[pID] < self.weight[qID]:
            self.id[pID] = qID
            self.weight[qID] += self.weight[pID]
        else:
            self.id[qID] = pID
            self.weight[pID] += self.weight[qID]







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
                    self.edges.append(Graph_edge(self.nodes[i], self.nodes[j], graph_data[i][j]))
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
                    print(value,weight)
                    self.nodes[i].add_adj(self.nodes[self.hash_map[value]])
                    self.edges.append(Graph_edge(self.nodes[i], self.nodes[self.hash_map[value]], weight))
                    self.edge_map[str(i) + "+" + str(self.hash_map[value])] = self.edges[-1]

    def output_to_matrix(self):
        ret = [[math.inf for j in range(len(self.nodes))] for i in range(len(self.nodes))]
        for edge in self.edges:
            i = edge.start.name
            j = edge.end.name
            ret[i][j] = edge.weight

        return ret

    def output_to_list(self):
        ret = [[node.data] for node in self.nodes]
        for edge in self.edges:
            i = edge.start.name
            j = edge.end.data
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
            self.edges.append(Graph_edge(node1, node2, weight))
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
            weight = edge.weight
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


    def prim(self):
        if self.is_empty():
            return
        v_new = [self.nodes[0]]
        v_hash = set([self.nodes[0].name])
        e_new = []
        v_length = len(self.nodes)
        current_length = 1
        while current_length != v_length:
            min_weight = math.inf
            min_node = None
            min_edge = None
            for node in v_new:
                for a_node in node.adj_node:
                    if a_node.name not in v_hash:
                        e12 = str(node.name) + "+" + str(a_node.name)
                        if e12 in self.edge_map:
                            edge = self.edge_map[e12]
                            if edge.weight < min_weight:
                                min_edge = edge
                                min_weight = edge.weight
                                min_node = a_node
            if not math.isinf(min_weight):
                v_new.append(min_node)
                current_length += 1
                e_new.append(min_edge)
                v_hash.add(min_node.name)

        return e_new

    def kruskal(self):
        union_node = WeightPathQuickUnionSet(len(self.nodes))
        edge_set = []
        edge_ret = []
        for edge in self.edges:
            heapq.heappush(edge_set, (edge.weight, edge.start.name,edge.end.name))
        total_count = len(self.nodes) - 1
        current_count = 0
        while edge_set and current_count < total_count:
            weight, start, end = heapq.heappop(edge_set)
            s1 = union_node.find(start)
            e1 = union_node.find(end)
            if s1 != e1:
                edge_ret.append(self.edge_map[str(start) + "+" + str(end)])
                union_node.union(s1, e1)
                current_count += 1

        return edge_ret
                
            

    def minimum_spaning_tree(self):
        if self.is_empty():
            return
        e_length = len(self.edges)
        v_length = len(self.nodes)
        if v_length * v_length < e_length * math.log(v_length):
            e_new = self.prim()
        else:
            e_new = self.kruskal()
        print("the minimum spaning tree")
        for edge in e_new:
            print("the edge: start node:{0}, end node:{1}, weight:{2}".format(edge.start.name, edge.end.name, edge.weight))
        print("---end---")

    def shortest_path(self):
        if self.is_empty():
            return

        count = 0
        while True:
            print("current number question: {0}".format(count))
            answer = "" 
            while True:
                answer = input("Do you want to find some shortest path? (y/n)").lower()
                if answer == "y" or answer == "yes" or answer == "n" or answer == "no":
                    break
            if answer == "n" or answer == "no":
                break
            self._floyd_warshall_print()
            print("")
            self._dijkstra_print()
            print("")
            self._bellman_ford_print()
            print("")
            count += 1
        
        
        
##        print("")
##        pprint.PrettyPrinter().pprint(A)
##        print("")
##        pprint.PrettyPrinter().pprint(path)

    def _dijkstra_print(self):
        print("the shortest path of the {0}".format(self.name))
        while True:
            try:
                start_node_name = int(input("you can input a start node name between {0} and {1}: ".format(0, len(self.nodes)-1)))
            except ValueError:
                print("that was no valid number. Try again...")
                continue
            if start_node_name >= 0 and start_node_name < len(self.nodes):
                break
            else:
                print("you should input a start node name between {0} and {1}, Try again...".format(0, len(self.nodes)-1))
                    
        node_ret, previous = self.dijkstra(self.nodes[start_node_name])

        end_node_name = 0
        while True:
            try:
                end_node_name = int(input("you can input a start node name between {0} and {1}: ".format(0, len(self.nodes)-1)))
            except ValueError:
                print("that was no valid number. Try again...")
                continue
            if end_node_name >= 0 and end_node_name < len(self.nodes):
                break
            else:
                print("you should input a start node name between {0} and {1}, Try again...".format(0, len(self.nodes)-1))

        s = deque()
        u = end_node_name
        #print(previous)
        while u != -1:
            s.appendleft(str(u))
            u = previous[u]
        print("the shortest path from {0} to {1}".format(start_node_name, end_node_name))
        print(" -> ".join(s))
        print("---the shortest path end---")
        
    def dijkstra(self, start_node):
        s = []
        q = []
        d = []
        previous = []
        visited = []
        for node in self.nodes:
            d.append(math.inf)
            previous.append(-1)
            q.append(node)
            visited.append(False)

        def _extract_min():
            ret = None
            min_value = math.inf
            for node in q:
                #print(d[node.name])
                if not visited[node.name] and d[node.name] < min_value:
                    min_value = d[node.name]
                    ret = node
            visited[ret.name] = True
            return ret
        
        d[start_node.name] = 0
        current_count = 0
        while current_count < len(self.nodes):
            u = _extract_min()
            s.append(u)
            for v in u.adj_node:
                e12 = str(u.name) + "+" + str(v.name)
                if e12 in self.edge_map:
                    weight = self.edge_map[e12].weight
                    if d[v.name] > d[u.name] + weight:
                        d[v.name] = d[u.name] + weight
                        previous[v.name] = u.name
            current_count += 1
        return s, previous

    def _floyd_warshall_print(self):
        A, path = self.floyd_warshall()
        visited = [False]*len(self.nodes)
        
        def _find_path(start, end, save_path):
            k = path[start][end]
            if path[start][end] == -1:
                if not visited[start]:
                    save_path.append(str(start))
                    visited[start] = True
                if not visited[end]:
                    save_path.append(str(end))
                    visited[end] = True
            else:
                _find_path(start, k, save_path)
                _find_path(k, end, save_path)
        print("the shortest path of the {0}".format(self.name))
        while True:
            try:
                start_node_name = int(input("you can input a start node name between {0} and {1}: ".format(0, len(self.nodes)-1)))
            except ValueError:
                print("that was no valid number. Try again...")
                continue
            if start_node_name >= 0 and start_node_name < len(self.nodes):
                break
            else:
                print("you should input a start node name between {0} and {1}, Try again...".format(0, len(self.nodes)-1))
                    
        end_node_name = 0
        while True:
            try:
                end_node_name = int(input("you can input a start node name between {0} and {1}: ".format(0, len(self.nodes)-1)))
            except ValueError:
                print("that was no valid number. Try again...")
                continue
            if end_node_name >= 0 and end_node_name < len(self.nodes):
                break
            else:
                print("you should input a start node name between {0} and {1}, Try again...".format(0, len(self.nodes)-1))

        s = []
        if math.isinf(A[start_node_name][end_node_name]):
            print("no shortest path from {0} to {1}".format(start_node_name, end_node_name))
        else:
            _find_path(start_node_name, end_node_name, s)
            print("the shortest path from {0} to {1}".format(start_node_name, end_node_name))
            print(" -> ".join(s))
        print("---the shortest path end---")
        
    def floyd_warshall(self):
        A = []
        path = []
        for i in range(len(self.nodes)):
            A.append([])
            path.append([])
            for j in range(len(self.nodes)):
                if i == j:
                    A[i].append(0)
                else:
                    A[i].append(math.inf)
                e12 = str(i) + "+" + str(j)
                if e12 in self.edge_map:
                    A[i][j] = self.edge_map[e12].weight
                path[i].append(-1)

        for k in range(len(self.nodes)):
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if A[i][j] > A[i][k] + A[k][j]:
                        A[i][j] = A[i][k] + A[k][j]
                        path[i][j] = k
        return A, path

    def _bellman_ford_print(self):
        print("the shortest path of the {0}".format(self.name))
        while True:
            try:
                start_node_name = int(input("you can input a start node name between {0} and {1}: ".format(0, len(self.nodes)-1)))
            except ValueError:
                print("that was no valid number. Try again...")
                continue
            if start_node_name >= 0 and start_node_name < len(self.nodes):
                break
            else:
                print("you should input a start node name between {0} and {1}, Try again...".format(0, len(self.nodes)-1))
                    
        node_ret, previous = self.bellman_ford(self.nodes[start_node_name])
        if not previous:
            print("Error exists!")
            return
        end_node_name = 0
        while True:
            try:
                end_node_name = int(input("you can input a start node name between {0} and {1}: ".format(0, len(self.nodes)-1)))
            except ValueError:
                print("that was no valid number. Try again...")
                continue
            if end_node_name >= 0 and end_node_name < len(self.nodes):
                break
            else:
                print("you should input a start node name between {0} and {1}, Try again...".format(0, len(self.nodes)-1))

        s = deque()
        u = end_node_name
        #print(previous)
        while u != -1:
            s.appendleft(str(u))
            u = previous[u]
        print("the shortest path from {0} to {1}".format(start_node_name, end_node_name))
        print(" -> ".join(s))
        print("---the shortest path end---")
        
    def bellman_ford(self, start_node):
        distance = [math.inf]*len(self.nodes)
        distance[start_node.name] = 0
        previous = [-1]*len(self.nodes)
        for i in range(len(self.nodes)-1):
            for node in self.nodes:
                s = node.name
                for a_node in node.adj_node:
                    t = a_node.name
                    e12 = str(s) + "+" + str(t)
                    if e12 in self.edge_map:
                        weight = self.edge_map[e12].weight
                        if distance[s] + weight < distance[t]:
                            distance[t] = distance[s] + weight
                            previous[t] = s
        for node in self.nodes:
            s = node.name
            for a_node in node.adj_node:
                t = a_node.name
                e12 = str(s) + "+" + str(t)
                if e12 in self.edge_map:
                    weight = self.edge_map[e12].weight
                    if distance[s] + weight < distance[t]:
                        print("the graph contain the negative weight ring")
                        return distance, None

        return distance, previous


    def two_match_dfs(self, node, check, matching):
        for a_node in node.adj_node:
            u = node.name
            v = a_node.name
            e12 = str(u) + "+" + str(v)
            if e12 in self.edge_map:
                if not check[v]:
                    check[v] = True
                    if matching[v] == -1 or self.two_match_dfs(self.nodes[matching[v]], check, matching):
                        matching[v] = u
                        matching[u] = v
                        return True

        return False

    def hurgarian_dfs(self):
        answer = 0
        matching = [-1]*len(self.nodes)
        check = [False]*len(self.nodes)
        for node in self.nodes:
            print(matching)
            u = node.name
            if matching[u] == -1:
                check = [False]*len(self.nodes)
                check[u] = True
                if self.two_match_dfs(node, check, matching):
                    answer += 1
        return answer, matching

    def hurgarian_bfs(self):
        Q = deque()
        prev = [0]*len(self.nodes) #remeber the path
        matching = [-1]*len(self.nodes)
        check = [-1]*len(self.nodes)
        answer = 0
        for i in range(len(self.nodes)):
            if matching[i] == -1:
                while Q:
                    Q.pop()
                Q.append(i)
                prev[i] = -1
                flag = False #Is find the path
                check[i] = i
                while Q and not flag:
                    u = Q[0]
                    for node_2 in self.nodes[u].adj_node:
                        if flag:
                            break
                        v = node_2.name
                        e12 = str(u) + "+" + str(v)
                        if e12 in self.edge_map:
                            if check[v] != i:
                                check[v] = i
                                Q.append(matching[v]) #question in here, is -1 can push into the queue
                                if matching[v] >= 0:
                                    prev[matching[v]] = u
                                else:
                                    flag = True
                                    d = u
                                    e = v
                                    while d != -1: #change the path to incrence the answer
                                        t = matching[d]
                                        matching[d] = e
                                        matching[e] = d
                                        d = prev[d]
                                        e = t
                    Q.pop()
                if matching[i] != -1:
                    answer += 1
        return answer, matching
                
                    

    def two_match_print(self):
        print("the maxmium two_matching of the {0}".format(self.name))
        answer, matching = self.hurgarian_bfs()
        visited = [False]*len(self.nodes)
        print("the maxmium two_matching number is {0}".format(answer))
        print("the matching pair is:")
        for index, value in enumerate(matching):
            if value != -1:
                print("{0} <-> {1}".format(index, value))
                visited[index] = True
                visited[value] = True

        print("---the end---")
                    

def test():
    data1 = [[math.inf, 1, 1, 1], [1, math.inf, 1, math.inf], [1, 1, math.inf, 1], [1, math.inf, 1, math.inf]]
    graph_1 = Graph(data1, data_type="matrix")
    print(graph_1)
    data_2 = graph_1.output_to_list()
    data_2 = [[1,[[2,2],[3,4],[4,1]]],[2,[[1,2],[4,3],[5,10]]],[3,[[1,4],[4,2],[6,5]]],[4,[[1,1],[2,3],[3,2],[5,2],[6,8],[7,4]]],
              [5,[[2,10],[4,2],[7,6]]],[6,[[3,5],[4,8],[7,1]]],[7,[[4,4],[5,6],[6,1]]]]
    graph_2 = Graph(data_2, data_type="list")
    print(graph_2)
    #graph_2.dfs()
    graph_2.bfs()
    print("")
    graph_2.minimum_spaning_tree()
    print("")
    graph_2.shortest_path()
    print("")
    graph_2.two_match_print()
    data_3 = [[1,[[2,1]]], [2, [[1,1],[3,1]]], [3, [[6,1],[2,1]]], [4,[[5,1]]], [5, [[4,1], [6,1],[8,1]]],[6, [[3,1],[5,1],[7,1],[9,1]]],
             [7, [[6,1]]], [8, [[5,1]]], [9, [[5,1]]]]
    graph_3 = Graph(data_3, data_type="list")
    print("")
    graph_3.two_match_print()

if __name__ == "__main__":
    test()
