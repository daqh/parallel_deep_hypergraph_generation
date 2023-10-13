from dgmh.utils import logAnalysis
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds, eigs
import snap
import networkx as nx

def k_decompose(k, hyperedges, size_limit=100000000):
    curr_nodes = []
    now = [0 for _ in range(k)]
    
    nodes = set([])
    _edges = {}
    edges = []
    
    for i, edge in enumerate(hyperedges):
        _nodes = []
        sz = len(edge)
        if sz > size_limit or sz < k: continue
        
        def find_node(p, idx):
            if idx == k:
                _nodes.append(tuple(now))
                return
            for nxt in range(p, sz - (k - idx) + 1):
                now[idx] = edge[nxt]
                find_node(nxt+1, idx+1)        
        
        find_node(0, 0)
        ext_nodes = []
        new_nodes = []
        _nodes = list(set(_nodes))
        for node in _nodes:
            if node in nodes:
                ext_nodes.append(node)
            else:
                new_nodes.append(node)
                _edges[node] = set([])
                nodes.add(node)
        ecnt = 0
        for i in range(len(ext_nodes)):
            for j in range(i+1, len(ext_nodes)):
                if ext_nodes[j] in _edges[ext_nodes[i]]: continue
                _edges[ext_nodes[i]].add(ext_nodes[j])
                _edges[ext_nodes[j]].add(ext_nodes[i])
                ecnt += 1
                edges.append((ext_nodes[i], ext_nodes[j]))
            for j in range(len(new_nodes)):
                _edges[ext_nodes[i]].add(new_nodes[j])
                _edges[new_nodes[j]].add(ext_nodes[i])
                edges.append((ext_nodes[i], new_nodes[j]))
                ecnt += 1
        for i in range(len(new_nodes)):
            for j in range(i+1, len(new_nodes)):
                _edges[new_nodes[i]].add(new_nodes[j])
                _edges[new_nodes[j]].add(new_nodes[i])
                edges.append((new_nodes[i], new_nodes[j]))
                ecnt += 1
    return list(nodes), edges

def analyze_gcc(nodes, edges):
    visited = set([])
    adjlist = {v: [] for v in nodes}

    for u, v in edges:
        adjlist[u].append(v)
        adjlist[v].append(u)

    que = [0 for _ in range(len(nodes))]
    max_cnt = 0
    for node in nodes:
        if node not in visited:
            head, tail, cnt = 0, 0, 0
            que[0] = node
            visited.add(node)
            while head >= tail:
                v = que[tail]
                tail += 1
                for nxt in adjlist[v]:
                    if nxt not in visited:
                        visited.add(nxt)
                        head += 1
                        que[head] = nxt
            max_cnt = max(max_cnt, tail)
    print("Size of the largest connected component: ", max_cnt)
    print("Connected Component: ", max_cnt / len(nodes))

def analyze_degrees(nodes, edges, filename):
    adjlist = {u: [] for u in nodes}
    for u, v in edges:
        adjlist[u].append(v)
        adjlist[v].append(u)
    degreeList = [len(v) for v in adjlist.values()]
    degrees, freq = np.unique(degreeList, return_counts=True)
    
    with open("./results/generated.{}_decomposition_degrees.txt".format(filename), "w") as f:
        for _x, _y in zip(degrees, freq):
            f.write(f'{_x} {_y}\n')
    
    start_idx = int(degrees[0] == 0)
    logAnalysis(degrees[start_idx:], freq[start_idx:],
                      "./plots/generated.{}_decomposition_degrees.png".format(filename),
                      "Degree", "Count", "Degrees")    

def analyze_singluar_values(nodes, edges, rank, filename):
    idxs = {}
    for i, k in enumerate(nodes): #enumerate(adjlist.keys()):
        idxs[k] = i
    rows, cols = [], []
    for u, v in edges:
        rows.append(idxs[u])
        cols.append(idxs[v])
        rows.append(idxs[v])
        cols.append(idxs[u])
    nnz = len(rows)
    incident_matrix = coo_matrix((np.ones(nnz), (rows, cols)), shape=(len(nodes), len(nodes)))
    
    _1, s, _2 = svds(incident_matrix.tocsc(), k=rank)
    s = sorted(s, reverse=True)
    
    with open("./results/generated.{}_decomposition_singular_values.txt".format(filename), "w") as f:
        for _x, _y in zip(np.arange(1,len(s)+1), s):
            f.write(f'{_x} {_y}\n')
            
    logAnalysis(np.arange(1,len(s)+1), s,
                      "./plots/generated.{}_decomposition_singular_values.png".format(filename),
                      "Rank", "Singular value", "Distribution of Singular Values")

def analyze_diameter(nodes, edges):
    G = snap.TUNGraph.New()
    idxs = {}
    for i, v in enumerate(nodes):
        idxs[v] = i
    for v in nodes:
        G.AddNode(idxs[v])
    for e in edges:
        G.AddEdge(idxs[e[0]], idxs[e[1]])
        G.AddEdge(idxs[e[1]], idxs[e[0]])
    DegToCCfV = snap.TFltPrV()
    print('Clustering Coefficient: ', snap.GetClustCf(G, -1))
    print('Effective Diameter: ', snap.GetBfsEffDiam(G, min(len(nodes), 10000), False))

def analyze_triangles(nodes, edges):
    G = nx.Graph()
    for v in nodes:
        G.add_node(v)
    for e in edges:
        G.add_edge(e[0], e[1])
    num_triangles = sum(nx.triangles(G).values()) / 3
    print('Number of triangles: ', num_triangles)

def analyze_k_decomposition(hyperedges, k, dataset_name):
    decomposed_type = [None, 'node', 'edge', 'triangle', '4clique']
    
    print(f"Analyzing {decomposed_type[k]}-decomposed graph...")
    projected_nodes, projected_edges = k_decompose(k, hyperedges, 7 if k > 1 else 25)
    print('Number of nodes: ', len(projected_nodes))
    print('Number of edges: ', len(projected_edges))

    analyze_gcc(projected_nodes, projected_edges)
    analyze_triangles(projected_nodes, projected_edges)
    analyze_degrees(projected_nodes, projected_edges, f"{dataset_name}_{k}")
    analyze_singluar_values(projected_nodes, projected_edges, 500, f"{dataset_name}_{k}")
    analyze_diameter(projected_nodes, projected_edges)
    print("")
