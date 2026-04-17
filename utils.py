#!/usr/bin/env python3

from collections import defaultdict, deque
from functools import partial, reduce
import json
import math
import multiprocessing
import random
import csv
import os
import glob

import dgl
import networkx as nx
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import torch                                                                          
import torch.nn as nn

# class for GEO enrichment analysis file
class GEOContext:
    def __init__(self, geo_id, pathway_genes):
        self.geo_id = geo_id                  # GEO id
        self.pathway_genes = pathway_genes    # TAKE pathway ids

        # biological context
        self.organ = None
        self.cell_type = None # Bulk for bulk RNAsq
        self.disease = None
        self.stimulus = None

        # numerica ids for embeddings
        self.organ_id = None
        self.cell_type_id = None
        self.disease_id = None
        self.stimulus_id = None

    @property
    def pathways(self):
        return list(self.pathway_genes.keys())

def compute_gene2pathway_dist(g, max_hops=3,):
    """
    Compute min-hop distance from each gene to each reachable pathway.

    Returns
    -------
    gene2pathway_dist : Dict[int, Dict[int, int]]
        gene_id -> { pathway_id : min_distance }
    """

    # --- Precompute direct memberships ---
    gene2direct_pathways = defaultdict(set)
    src, dst = g.edges(etype='in_pathway')  # gene -> pathway
    take_mask = g.nodes['pathway'].data['take_mask']
    for g_id, p_id in zip(src.tolist(), dst.tolist()):
        if not take_mask[p_id]:
            continue
        gene2direct_pathways[g_id].add(p_id)

    # --- Build PPI adjacency (gene-gene) ---
    ppi_src, ppi_dst = g.edges(etype='ppi')
    num_genes = g.num_nodes('gene')

    adj = [[] for _ in range(num_genes)]
    for u, v in zip(ppi_src.tolist(), ppi_dst.tolist()):
        adj[u].append(v)
        adj[v].append(u) # undirected PPI

    # --- Main result ---
    gene2pathway_dist = dict()

    # --- BFS per gene ---
    for seed_gene in range(num_genes):
        dist_map = dict()  # pathway_id -> min_dist
        visited = set([seed_gene])
        q = deque([(seed_gene, 0)])

        # distance 0
        for p in gene2direct_pathways.get(seed_gene, []):
            dist_map[p] = 0

        while q:
            gene, d = q.popleft()
            if d == max_hops:
                continue

            for nbr in adj[gene]:
                if nbr in visited:
                    continue
                visited.add(nbr)
                nd = d + 1

                # record pathways of neighbor
                for p in gene2direct_pathways.get(nbr, []):
                    if p not in dist_map or nd < dist_map[p]:
                        dist_map[p] = nd

                q.append((nbr, nd))

        if dist_map:
            gene2pathway_dist[seed_gene] = dist_map

    return gene2pathway_dist

def get_leaf_descendants(g, pathway_id):
    """
    Return all TAKE leaf descendants of a given pathway
    """
    take_mask = g.nodes['pathway'].data['take_mask']

    visited = set()
    stack = [pathway_id]

    leaf_nodes = []

    while stack:
        p = stack.pop()
        if p in visited:
            continue
        visited.add(p)

        # find children
        children = g.successors(p, etype='parent_of').tolist()
        if len(children) == 0:
            # leaf node
            if take_mask[p]:
                leaf_nodes.append(p)
        else:
            stack.extend(children)

    return leaf_nodes

def build_leaf_descendant_cache(g):
    cache = {}

    num_pathways = g.num_nodes('pathway')

    for p in range(num_pathways):
        cache[p] = get_leaf_descendants(g,p)

    return cache

def precompute_hierarchy_levels(g):
    id2pathway = g.graph_data['id2pathway']
    # 1. Find the Roots (In 'child_of', roots are nodes that are never 'src')
    num_pathways = g.num_nodes('pathway')
    all_pathway_ids = set(range(num_pathways))

    src, dst = g.edges(etype='child_of')
    all_nodes = set(range(g.num_nodes('pathway')))
    participating_nodes = set(src.tolist()) | set(dst.tolist())
    isolated_pathways = list(all_nodes - participating_nodes)
    iso_pathways = [id2pathway[i] for i in isolated_pathways[:100]]
    print(f"  - Isolated nodes: {', '.join(iso_pathways)}")

    src_nodes = set(src.tolist())
    roots = list((all_nodes - src_nodes) | (all_nodes - participating_nodes))

    root_pathways = [id2pathway[r] for r in roots[:100]]
    print(f"  - Root nodes: {', '.join(root_pathways)}")
    
    # 2. Assign the maximum distance from root to each node
    # This handles your 'a', 'b', 'c' contradiction
    node_to_level = {root: 0 for root in roots}
    
    # We use a simple loop to propagate levels downward
    # (Using 'parent_of' edges here is easier: Parent -> Child)
    p_src, p_dst = g.edges(etype='parent_of')
    adj = {s.item(): [] for s in p_src}
    for s, d in zip(p_src, p_dst):
        adj[s.item()].append(d.item())

    # Breadth-First update to ensure every node gets its MAX depth
    queue = list(roots)
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, []):
            # Level of child is Parent + 1
            new_level = node_to_level[u] + 1
            if v not in node_to_level or new_level > node_to_level[v]:
                node_to_level[v] = new_level
                queue.append(v)
            
    # 3. Group nodes by level and REVERSE them for Bottom-Up aggregation
    max_level = max(node_to_level.values()) if node_to_level else 0
    levels = [[] for _ in range(max_level + 1)]
    for node, lvl in node_to_level.items():
        levels[lvl].append(node)
        
    # We return reversed levels: [Deepest Leaves, ..., Roots]
    return [torch.tensor(lvl) for lvl in reversed(levels)]

def precompute_gene_relative_depths(g):
    """
    Calculates the relative depth for every 'has_gene' edge.
    Depth 0 = The most specific (deepest) pathway(s) a gene belongs to.
    Depth N = How many steps a parent is above that gene's deepest pathway.
    """
    # 1. Build the Hierarchy Graph (Parent -> Child)
    # We use this to find descendants and calculate path lengths
    ps, pd = g.edges(etype='parent_of')
    hx = nx.DiGraph()

    num_pathways = g.num_nodes('pathway')
    hx.add_nodes_from(range(num_pathways))

    hx.add_edges_from(zip(ps.tolist(), pd.tolist()))

    # Identify global leaves
    global_leaves = {n for n, d in hx.out_degree() if d == 0}

    # 2. Map Gene -> All its associated Pathways
    # p_src: Pathway IDs, g_dst: Gene IDs
    p_src, g_dst = g.edges(etype='has_gene')
    p_src_list = p_src.tolist()
    g_dst_list = g_dst.tolist()
    
    # Group pathway indices by gene_id for faster lookup
    gene_to_edge_indices = {}
    for i, gene_id in enumerate(g_dst_list):
        if gene_id not in gene_to_edge_indices:
            gene_to_edge_indices[gene_id] = []
        gene_to_edge_indices[gene_id].append(i)

    # 3. Prepare the result tensor (same size as the number of 'has_gene' edges)
    num_edges = g.num_edges('has_gene')
    edge_rel_depth = torch.zeros(num_edges, dtype=torch.long)
    edge_leaf_type = torch.zeros(num_edges, dtype=torch.long)

    print("Calculating Gene-Specific Relative Depths...")
    # 4. Iterate through each gene to find its "Local Leaves"
    for gene_id, edge_indices in tqdm(gene_to_edge_indices.items()):
        # Get the actual pathway IDs associated with this gene
        path_ids = [p_src_list[idx] for idx in edge_indices]
        path_set = set(path_ids)
        
        # A pathway is a 'Local Leaf' if none of its descendants in the 
        # hierarchy are also associated with this specific gene.
        local_leaves = []
        for p in path_ids:
            # Get all nodes reachable from p in the hierarchy
            descendants = nx.descendants(hx, p)
            # If no descendant is in the gene's pathway list, p is a local leaf
            if not any(d in path_set for d in descendants):
                local_leaves.append(p)

        local_leaf_set = set(local_leaves)
        
        # 5. Calculate Distance from each Parent to the nearest Local Leaf
        for idx in edge_indices:
            p_id = p_src_list[idx]

            # A. Determine leaf type
            if p_id in local_leaf_set:
                if p_id in global_leaves:
                    edge_leaf_type[idx] = 0 # REAL LEAF
                else:
                    edge_leaf_type[idx] = 1 # LOCAL LEAF
                edge_rel_depth[idx] = 0
            else:
                edge_leaf_type[idx] = 2 # PARENT
                # Find shortest path from this parent to any of the gene's local leaves
                dists = []
                for leaf in local_leaves:
                    try:
                        dists.append(nx.shortest_path_length(hx, source=p_id, target=leaf))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                
                # If a path exists, take the minimum distance (closest leaf)
                # If no path (disjoint hierarchy), default to a safe value like 0 or 10
                edge_rel_depth[idx] = min(dists) if dists else 0

    # 5. Attach to the graph
    g.edges['has_gene'].data['rel_depth'] = edge_rel_depth
    g.edges['has_gene'].data['leaf_type'] = edge_leaf_type


def verify_precomputed_hierarchy(gene_symbol, g):
    # 1. Setup Lookups
    id2pathway = g.graph_data['id2pathway']
    id2gene = g.graph_data['id2gene']
    pathway2id = g.graph_data['pathway2id']
    gene2id = g.graph_data['gene2id']
    
    gene_id = gene2id[gene_symbol]
    
    if gene_id is None:
        print(f"Gene '{gene_symbol}' not found.")
        return

    # 2. Map Leaf Type Integers to Labels
    type_map = {
        0: "REAL LEAF",
        1: "LOCAL LEAF",
        2: "PARENT"
    }

    # 3. Find the edges for this specific gene in the 'has_gene' etype
    # p_src: Pathway IDs, g_dst: Gene IDs
    p_src, g_dst = g.edges(etype='has_gene')
    
    # Get indices where the destination gene is our target gene
    edge_indices = (g_dst == gene_id).nonzero(as_tuple=True)[0]

    if len(edge_indices) == 0:
        print(f"No 'has_gene' edges found for {gene_symbol}.")
        return

    # 4. Extract precomputed data for these specific edges
    # DGL stores edge data in the same order as the edges themselves
    pre_depths = g.edges['has_gene'].data['rel_depth'][edge_indices]
    pre_types = g.edges['has_gene'].data['leaf_type'][edge_indices]
    pathway_ids = p_src[edge_indices]

    # 5. Format and Print
    print(f"\n{'='*70}")
    print(f" PRECOMPUTED DATA CHECK FOR GENE: {gene_symbol}")
    print(f"{'='*70}")
    
    results = []
    for i in range(len(edge_indices)):
        p_id = pathway_ids[i].item()
        depth = pre_depths[i].item()
        l_type = pre_types[i].item()
        
        results.append({
            'name': id2pathway[p_id],
            'type': type_map.get(l_type, "UNKNOWN"),
            'depth': depth
        })

    # Sort: Leaves first, then by depth
    results.sort(key=lambda x: (0 if "LEAF" in x['type'] else 1, x['depth']))

    print(f"{'TYPE':<12} | {'DEPTH':<5} | {'PATHWAY NAME'}")
    print("-" * 75)
    for res in results:
        print(f"{res['type']:<12} | {res['depth']:<5} | {res['name']}")


def trace_gene_local_hierarchy(gene_symbol, g):
    """
    Diagnostic tool to print the hierarchy of a specific gene.
    Identifies 'Local Leaves' and calculates Relative Depth for all parents.
    """
    # 1. Setup Lookups
    id2pathway = g.graph_data['id2pathway']
    id2gene = g.graph_data['id2gene']
    pathway2id = g.graph_data['pathway2id']
    gene2id = g.graph_data['gene2id']

    gene_id = gene2id[gene_symbol]
    
    if gene_id is None:
        print(f"Gene '{gene_symbol}' not found in the provided gene2id mapping.")
        return

    # 2. Build Hierarchy Graph (Parent -> Child)
    ps, pd = g.edges(etype='parent_of')
    hx = nx.DiGraph()
    hx.add_edges_from(zip(ps.tolist(), pd.tolist()))

    global_leaves = {n for n, d in hx.out_degree() if d == 0}

    # 3. Find all pathways associated with this gene
    # has_gene: [Pathway -> Gene]
    p_src, g_dst = g.edges(etype='has_gene')
    associated_indices = (g_dst == gene_id).nonzero(as_tuple=True)[0]
    associated_p_ids = p_src[associated_indices].tolist()
    path_set = set(associated_p_ids)

    if not associated_p_ids:
        print(f"No pathways found for gene {gene_symbol} in the graph edges.")
        return

    # 4. Identify Local Leaves
    # A local leaf is a pathway where no descendant is ALSO in path_set
    local_leaves = []
    for p in associated_p_ids:
        descendants = nx.descendants(hx, p)
        if not any(d in path_set for d in descendants):
            local_leaves.append(p)

    # 5. Calculate and Print Results
    print(f"\n{'='*60}")
    print(f" HIERARCHY TRACE FOR GENE: {gene_symbol} (ID: {gene_id})")
    print(f"{'='*60}")
    print(f"Found {len(associated_p_ids)} associated pathways.")
    print(f"Identified {len(local_leaves)} Local Leaf/Leaves (Specificity Bases).\n")

    # Sort pathways by their distance to the nearest local leaf for better reading
    results = []
    for p_id in associated_p_ids:
        if p_id in local_leaves:
            is_global = p_id in global_leaves
            if is_global:
                results.append({
                    'name': id2pathway[p_id],
                    'type': 'REAL LEAF',
                    'rel_depth': 0,
                    'via': 'Self'
                })
            else:
                results.append({
                    'name': id2pathway[p_id],
                    'type': 'LOCAL LEAF',
                    'rel_depth': 0,
                    'via': 'Self'
                })
        else:
            # Find distance to nearest local leaf
            shortest_dist = float('inf')
            closest_leaf_name = "None"
            for leaf in local_leaves:
                try:
                    d = nx.shortest_path_length(hx, source=p_id, target=leaf)
                    if d < shortest_dist:
                        shortest_dist = d
                        closest_leaf_name = id2pathway[leaf]
                except nx.NetworkXNoPath:
                    continue
            
            results.append({
                'name': id2pathway[p_id],
                'type': 'PARENT',
                'rel_depth': shortest_dist if shortest_dist != float('inf') else "N/A",
                'via': closest_leaf_name
            })

    # Sort results: Leaves first, then by relative depth
    results.sort(key=lambda x: (0 if x['type'] == 'LOCAL LEAF' else 1, 
                                x['rel_depth'] if isinstance(x['rel_depth'], int) else 99))

    # Print Formatted Table
    print(f"{'TYPE':<12} | {'DEPTH':<5} | {'PATHWAY NAME':<50} | {'VIA LEAF'}")
    print("-" * 100)
    for res in results:
        depth_display = str(res['rel_depth'])
        print(f"{res['type']:<12} | {depth_display:<5} | {res['name'][:50]:<50} | {res['via'][:30]}")



# load the reactome hierarchical pathways
def load_reactome_mux_graph(ppi_file, pathway_gene_file, pathway_rel_file):
    """
    Load a multiplex heterogeneous graph for Reactome pathway, gene, and PPI data.
    This will create a DGL heterograph for gene–gene PPIs, gene–pathway memberships, and pathway hierarchy.
    """
    # Step 1: Load Gene-PPI data (merged_signaling_network_unique.tsv)
    df_ppi = pd.read_csv(ppi_file, sep='\t')
    ppi_edges = [(row['gene1'], row['gene2']) for _, row in df_ppi.iterrows()]

    # Step 2: Load Pathway-Gene membership (reactome_pathway_gene.csv)
    df_pathways = pd.read_csv(pathway_gene_file)
    gene_to_pathway = defaultdict(list)
    for _, row in df_pathways.iterrows():
        pathway_id = row['Pathway_ID']
        genes = [g.strip()for g in str(row['Signaling_Genes']).split(',') if g.strip()]
        for gene in genes:
            gene_to_pathway[gene].append(pathway_id)

    # Step 3: Load Pathway Relations (ReactomePathwaysRelation.MMU.gene.txt)
    df_relations = pd.read_csv(pathway_rel_file, sep='\t', header=None, names=['parent', 'child'])

    # Step 4: Create a mapping for Gene IDs and Pathway IDs
    gene2id = {gene: idx for idx, gene in enumerate(gene_to_pathway.keys())}
    pathway_ids_from_rel = set(df_relations['parent']).union(set(df_relations['child']))
    pathway_ids_from_genes = set()
    for pathways in gene_to_pathway.values():
        pathway_ids_from_genes.update(pathways)
    all_pathways = pathway_ids_from_rel.union(pathway_ids_from_genes)
    pathway2id = {pathway: idx for idx, pathway in enumerate(sorted(all_pathways))}

    # the inverse mappings
    id2gene = {v: k for k, v in gene2id.items()}
    id2pathway = {v: k for k, v in pathway2id.items()}

    num_genes = len(gene2id)
    num_pathways = len(pathway2id)
    
    # Step 5: Create edges for gene-pathway and pathway-gene relations
    in_pathway_edges = []
    has_gene_edges = []
    for gene, pathways in gene_to_pathway.items():
        gene_idx = gene2id[gene]
        for pathway in pathways:
            pathway_idx = pathway2id[pathway]
            in_pathway_edges.append((gene_idx, pathway_idx))
            has_gene_edges.append((pathway_idx, gene_idx))

    parent_edges = []
    child_edges = []
    for _, row in df_relations.iterrows():
        u, v = pathway2id[row['parent']], pathway2id[row['child']]
        parent_edges.append((u, v))
        child_edges.append((v, u))

    # Create bidirectional & unique PPI
    unique_interactions = set()
    for _, row in df_ppi.iterrows():
        g1, g2 = row['gene1'], row['gene2']
        if g1 in gene2id and g2 in gene2id:
            u, v = gene2id[g1], gene2id[g2]
            if u == v:
                continue

            interaction = tuple(sorted((u,v)))
            unique_interactions.add(interaction)

    ppi_edges_ids = []
    for u, v in unique_interactions:
        ppi_edges_ids.append((u, v))
        ppi_edges_ids.append((v, u))

    take_mask = torch.zeros(num_pathways, dtype=torch.bool)

    for _, row in df_pathways.iterrows():
        pid = row['Pathway_ID']
        if row['TAKE'] == 1:
            take_mask[pathway2id[pid]] = True

    gene2pathways = defaultdict(set)
    for _, row in df_pathways.iterrows():
        pid = pathway2id[row['Pathway_ID']]
        if not take_mask[pid]:
            continue
        genes = [g.strip() for g in str(row['Signaling_Genes']).split(',') if g.strip()]
        for gene in genes:
            if gene in gene2id:
                gene2pathways[gene2id[gene]].add(pid)

    ppi_edge_pathway_mask = torch.zeros(len(ppi_edges_ids), num_pathways, dtype=torch.bool)
    for e, (u, v) in enumerate(ppi_edges_ids):
        common = gene2pathways[u] & gene2pathways[v]
        for p in common:
            ppi_edge_pathway_mask[e, p] = True

    # Step 6: Build the DGL heterograph
    g = dgl.heterograph({
        ('gene', 'ppi', 'gene'): ppi_edges_ids,
        ('pathway', 'parent_of', 'pathway'): parent_edges,
        ('pathway', 'child_of', 'pathway'): child_edges,
        ('gene', 'in_pathway', 'pathway'): in_pathway_edges,
        ('pathway', 'has_gene', 'gene'): has_gene_edges
        }, num_nodes_dict={'gene': num_genes, 'pathway': num_pathways})

    # Step 7: Assign features (currently random; could be replaced with real data)

    feat_dim = 128

    g.nodes['gene'].data['feat'] = torch.randn(num_genes, feat_dim)
    g.nodes['pathway'].data['feat'] = torch.randn(num_pathways, feat_dim)

    # mask for leaf pathway
    g.nodes['pathway'].data['take_mask'] = take_mask
    # mask for one ppi belonging to one pathway
    g.edges['ppi'].data['pathway_mask'] = ppi_edge_pathway_mask

    gene2pathway_dist = compute_gene2pathway_dist(g, max_hops=3)

    g.graph_data = {}
    g.graph_data['gene2pathway_dist'] = gene2pathway_dist

    # store pathway's chidren
    g.graph_data['leaf_descendants'] = build_leaf_descendant_cache(g)

    # store pathway genes
    g.graph_data['pathway_genes'] = {
        pw: g.successors(pw, etype='has_gene').tolist()
        for pw in range(g.num_nodes('pathway'))
    }
    g.graph_data['leaf_pathway_genes'] = {
        pw: g.successors(pw, etype='has_gene').tolist()
        for pw in range(g.num_nodes('pathway'))
        if take_mask[pw]
    }
    # store gene pathways
    gene2pathways = {
        gene: list(pathways)
        for gene, pathways in gene2pathways.items()
    }
    g.graph_data['gene2pathways'] = gene2pathways

    # add the two id maps
    g.graph_data['id2pathway'] = id2pathway
    g.graph_data['id2gene'] = id2gene
    g.graph_data['pathway2id'] = pathway2id
    g.graph_data['gene2id'] = gene2id

    # Step 8. Pre-compute Batched Pathway Subgraphs
    subgraph_list = []
    pathway_indices = []
    for p in range(num_pathways):
        if not take_mask[p]:
            continue

        eids = torch.nonzero(ppi_edge_pathway_mask[:, p], as_tuple=True)[0]

        if eids.numel() > 0:
            subp = dgl.edge_subgraph(g['ppi'], eids)
            subgraph_list.append(subp)
            pathway_indices.append(p)

    if subgraph_list:
        # This graph contains all leaf pathway subgraphs
        g.graph_data['batched_ppi_graph'] = dgl.batch(subgraph_list)
        g.graph_data['pathway_indices'] = torch.tensor(pathway_indices)
        # Maps pathway_id -> index in the batch
        pathway_to_batch_idx = torch.full((num_pathways,), -1, dtype=torch.long)
        pathway_to_batch_idx[pathway_indices] = torch.arange(len(pathway_indices))
        g.graph_data['pathway_to_batch_idx'] = pathway_to_batch_idx

    # Add hierarchy ordered tree
    g.levels = precompute_hierarchy_levels(g)
    precompute_gene_relative_depths(g)
    
    #verify_hierarchy_levels(g)
    #print_hierarchy_tree(g)

    # Return the graph and the gene/pathway ID mappings
    return g, gene2id, pathway2id

def verify_hierarchy_levels(g):
    """
    g: DGL graph with g.levels and g.nodes['pathway'].data['take_mask']
    id2pathway: list or dict mapping DGL index -> R-MMU-ID string
    """
    # 1. Get Metadata from Graph
    id2pathway = g.graph_data['id2pathway']
    take_mask = g.nodes['pathway'].data['take_mask'].cpu().numpy()

    # We want to iterate from Level 0 (Roots) down to the deepest leaves
    # g.levels is [Deepest_Leaves, ..., Level 1, Level 0]
    root_to_leaf_levels = list(reversed(g.levels))

    # Map for Parent-Child validation
    # node -> level_index
    node_to_lvl_num = {}
    for l_idx, nodes in enumerate(root_to_leaf_levels):
        for n in nodes.tolist():
            node_to_lvl_num[n] = l_idx

    print(f"\n{'LEVEL':<6} | {'IDX':<5} | {'R-MMU-ID':<15} | {'TAKE':<5} | {'INTEGRITY'}")
    print("-" * 65)

    errors = 0
    for l_num, nodes in enumerate(root_to_leaf_levels):
        for node_idx in sorted(nodes.tolist()):
            rmmu_id = id2pathway[node_idx]
            is_take = "TAKE" if take_mask[node_idx] == 1 else "----"

            # --- Parent-Child Check ---
            # In 'parent_of', the child must have a level HIGHER than the parent
            _, children = g.out_edges(node_idx, etype='parent_of')
            status = "OK"
            for c in children.tolist():
                child_lvl = node_to_lvl_num.get(c)
                if child_lvl is not None and child_lvl <= l_num:
                    status = "❌ ERR"
                    errors += 1

            indent = "  " * l_num
            print(f"L_{l_num:<4} | {node_idx:<5} | {rmmu_id:<15} | {is_take:<5} | {indent}{status}")

        if len(nodes) > 0:
            print("-" * 20)

    print(f"\nVerification Complete. Total Integrity Errors: {errors}")
    if errors == 0:
        print("✅ Success: Every parent is positioned strictly above its children.")
    else:
        print("⚠️ Warning: Hierarchy level contradiction detected.")

def print_hierarchy_tree(g, max_depth=10):
    """
    Prints a visual tree of the hierarchy with Level and TAKE info.
    """
    root_list = list(reversed(g.levels))
    root_nodes = root_list[0].tolist()

    id2pathway = g.graph_data['id2pathway']
    take_mask = g.nodes['pathway'].data['take_mask'].cpu().numpy()

    # We need a node -> level lookup for the printout
    node_to_lvl = {}
    for l_idx, nodes in enumerate(reversed(g.levels)):
        for n in nodes.tolist():
            node_to_lvl[n] = l_idx

    def _recurse(node, prefix="", is_last=True, current_depth=0):
        if current_depth > max_depth:
            return

        # Get Metadata
        rmmu_id = id2pathway[node]
        lvl = node_to_lvl.get(node, "?")
        flag = " [TAKE]" if take_mask[node] == 1 else ""

        # Format the line
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{rmmu_id} (L_{lvl}){flag}")

        # Find children using 'parent_of' edges
        _, children = g.out_edges(node, etype='parent_of')
        child_list = sorted(children.tolist())

        # Update prefix for next level
        new_prefix = prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(child_list):
            _recurse(child, new_prefix, i == len(child_list) - 1, current_depth + 1)

    print("\nVisual Hierarchy Tree (Root -> Leaf)")
    print("Format: R-MMU-ID (Calculated_Level) [TAKE_Flag]")
    print("-" * 50)

    for i, root in enumerate(sorted(root_nodes)):
        _recurse(root, is_last=(i == len(root_nodes) - 1))



def verify_batch_alignment(g):
    """
    Sanity check to ensure pathway IDs align with the nodes in the batched graph.
    """
    print("\n--- [SANITY CHECK: BATCH ALIGNMENT] ---")
    
    # 1. Get the indices and the counts
    batched_g = g.graph_data['batched_ppi_graph']
    pathway_indices = g.graph_data['pathway_indices']
    id2pathway = g.graph_data['id2pathway']
    id2gene = g.graph_data['id2gene']

    nodes_per_pathway = batched_g.batch_num_nodes()
    
    # 2. Perform the repeat_interleave (The logic we are testing)
    pathway_ids_for_nodes = torch.repeat_interleave(
        pathway_indices, 
        nodes_per_pathway
    )

    global_gene_nids = batched_g.ndata[dgl.NID]

    # 3. Print mapping for the first 3 subgraphs
    current_node_ptr = 0
    for i in range(min(3, len(pathway_indices))):
        p_id = pathway_indices[i].item()
        p_name = id2pathway[p_id]
        num_n = nodes_per_pathway[i].item()
        
        sample_size = min(4, num_n)
        sample_global_nids = global_gene_nids[current_node_ptr : current_node_ptr + sample_size]
        sample_gene_names = [id2gene[nid.item()] for nid in sample_global_nids]

        print(f"Pahtway Index {i}:")
        print(f"    - Reactome ID: {p_name}")
        print(f"    - Total Genes: {num_n}")
        print(f"    - Sample Genes in Batch: {', '.join(sample_gene_names)} ...")

        # Slice the mapping for this specific "island" (subgraph)
        node_slice = pathway_ids_for_nodes[current_node_ptr : current_node_ptr + num_n]
        if (node_slice == p_id).all():
            print(f"     Logic Verification: All nodes correctly map to index {p_id}")
        else:
            print(f"    Error: mismatch in pathway indexing!")
        
        current_node_ptr += num_n

    print("--- [ALIGNMENT VERIFIED: SUCCESS] ---\n")

def verify_gnn_isolation(g, model_layer):
    """
    Verifies that the GNN layer preserves pathway isolation in the batched graph.
    """
    device = next(model_layer.parameters()).device
    
    id2pathway = g.graph_data['id2pathway']
    id2gene = g.graph_data['id2gene']
    
    batched_g = g.graph_data['batched_ppi_graph'].to(device)
    pathway_indices = g.graph_data['pathway_indices']
    nodes_per_p = batched_g.batch_num_nodes()
    global_nids = batched_g.ndata[dgl.NID].to(device)

    print("\n" + "="*50)
    print("  GNN ISOLATION & MAPPING TEST")
    print("="*50)

    # 1. Pick a "Source" Pathway (e.g., the first one) and a "Target" Gene
    # Let's find a gene in Pathway 0 that actually has an edge
    start_ptr = 0
    end_ptr = nodes_per_p[0].item()

    # Create a zero-feature matrix for all 60k active nodes
    feat_dim = 128 # Small dim for testing
    test_h = torch.zeros(len(global_nids), feat_dim).to(device)

    # Inject a unique signal into the first gene of the first pathway
    test_h[0, :] = 100.0
    source_gene_id = global_nids[0].item()
    source_pathway_id = pathway_indices[0].item()

    source_gene_name = id2gene[source_gene_id]
    source_pathway_name = id2pathway[source_pathway_id]

    # 2. Run the GNN layer
    # We use a GraphConv or GAT layer here
    with torch.no_grad():
        # DGL's layer will only move features across edges in 'batched_g'
        out_h = model_layer(batched_g, test_h)
        if isinstance(out_h, torch.Tensor) and out_h.dim() == 3: # Handle GAT multi-head
            out_h = out_h.mean(1)

    # 3. Check for Leakage
    # Any node with a value > 0 received the signal
    active_indices = torch.where(out_h.sum(dim=1) > 0)[0]

    # Determine which pathways these active nodes belong to
    pathway_ids_for_nodes = torch.repeat_interleave(
        pathway_indices.to(device), nodes_per_p.to(device)
    )
    active_pathways = pathway_ids_for_nodes[active_indices].unique().tolist()
    active_pathways_names = [id2pathway[idx] for idx in active_pathways]

    print(f"Source: Gene {source_gene_name} in Pathway {source_pathway_name}")
    print(f"Active Pathways after GNN: {len(active_pathways)}: {active_pathways_names}")

    # VERIFICATION 1: Is the signal only in the source pathway?
    if len(active_pathways) == 1 and active_pathways[0] == source_pathway_id:
        print(f"  [✓] ISOLATION SUCCESS: Signal strictly contained in Pathway {source_pathway_name}")
        recipient_genes = [id2gene[global_nids[idx].item()] for idx in active_indices[:10]]
        print(f"    - Genes that received signal: {', '.join(recipient_genes)}...")
    else:
        print(f"  [X] LEAKAGE ERROR: Signal spread to pathways {active_pathways_names}")

    # VERIFICATION 2: Check global mapping
    # Ensure the genes that became 'active' are actually PPI neighbors of source_gene_id
    # in the original Reactome data for that specific pathway
    recipient_genes = [id2gene[global_nids[idx].item()] for idx in active_indices[:20]]
    print(f"  - Nodes in batch receiving signal: {', '.join(recipient_genes)}")

    print("="*50 + "\n")


def verify_gnn_stress_test(g, model_layer, num_test_pathways=5):
    """
    Randomized Stress Test:
    Injects unique signals into N random pathways and ensures 
    zero cross-talk and perfect mapping.
    """
    device = next(model_layer.parameters()).device
    
    batched_g = g.graph_data['batched_ppi_graph'].to(device)
    pathway_indices = g.graph_data['pathway_indices']
    nodes_per_p = batched_g.batch_num_nodes()
    global_nids = batched_g.ndata[dgl.NID].to(device)
    
    id2pathway = g.graph_data['id2pathway']
    id2gene = g.graph_data['id2gene']

    test_idx = 123
    global_id = batched_g.ndata[dgl.NID][test_idx].item()

    feat = g.nodes['gene'].data['feat']
    input_proj = nn.Linear(128, 128)
    input_norm = nn.LayerNorm(128)
    h = input_proj(feat)
    h = input_norm(h)

    global_nids = batched_g.ndata[dgl.NID]
    h_batched = h[global_nids.to(h.device)]
    
    val_in_h_batched = h_batched[test_idx]
    val_in_global_h = h[global_id]

    difference = torch.abs(val_in_h_batched - val_in_global_h).sum().item()
    print(f"Index Mapping Error: {difference}")

    # Get feature dimension
    feat_dim = model_layer.fc.in_features if hasattr(model_layer, 'fc') else 128
    
    # 1. Select random pathways to infect with a signal
    all_p_indices = list(range(len(pathway_indices)))
    sampled_p_indices = random.sample(all_p_indices, min(num_test_pathways, len(all_p_indices)))
    
    test_h = torch.zeros(len(global_nids), feat_dim).to(device)
    
    # Track the ground truth: {batch_index: (pathway_idx, marker_value)}
    infection_targets = {}
    current_ptr = 0
    
    print("\n" + "="*70)
    print(f"  GNN MULTI-PATHWAY STRESS TEST ({len(sampled_p_indices)} Pathways)")
    print("="*70)

    # 2. Inject Unique Markers
    batch_num_nodes = batched_g.batch_num_nodes()
    cum_nodes = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch_num_nodes, dim=0)])

    #for i, p_idx in enumerate(all_p_indices):
    for i, p_list_idx in enumerate(sampled_p_indices):
        #num_nodes = nodes_per_p[i].item()
        marker = 1 + (2*i)
        start_idx = cum_nodes[p_list_idx].item()

        test_h[start_idx, :] = marker

        infection_targets[p_list_idx] = {
            'marker': marker,
            'pathway_name': id2pathway[pathway_indices[p_list_idx].item()],
            'start_idx': start_idx,
            'end_idx': cum_nodes[p_list_idx+1].item()
        }

        #if i in sampled_p_indices:
            # Use powers of 10 as markers: 1.0, 10.0, 100.0...
        #    marker = 10.0**len(infection_targets)
            # Infect the first node of this pathway
        #    test_h[current_ptr, :] = marker
        #    infection_targets[i] = {
        #        'marker': marker,
        #        'pathway_name': id2pathway[pathway_indices[i].item()],
        #        'start_idx': current_ptr,
        #        'end_idx': current_ptr + num_nodes
        #    }
        #current_ptr += num_nodes

    # 3. Run GNN
    with torch.no_grad():
        out_h = model_layer(batched_g, test_h)
        if isinstance(out_h, torch.Tensor) and out_h.dim() == 3:
            out_h = out_h.mean(1)

    # 4. Verification Loop
    overall_success = True
    current_ptr = 0

    for i in range(len(pathway_indices)):
        num_nodes = nodes_per_p[i].item()
        p_name = id2pathway[pathway_indices[i].item()]
        
        # Define the strict boundaries for this "Island"
        island_slice = out_h[current_ptr : current_ptr + num_nodes]
        
        if i in sampled_p_indices:
            expected_marker = infection_targets[i]['marker']
            island_slice = out_h[current_ptr : current_ptr + num_nodes]

            print(f"    - Detailed Gene Breakdown for {p_name}:")

            island_global_nids = global_nids[current_ptr : current_ptr + num_nodes]
            for j in range(num_nodes):
                gene_name = id2gene[island_global_nids[j].item()]
                gene_val = island_slice[j].mean().item()

                tag = "[SOURCE]" if j == 0 else ""

                if gene_val > 1e-5 or j < 10:
                    print(f"    * {gene_name:10}: {gene_val:10.2f} {tag}")
            
            # 1. Verify the signal is present in its own island
            # We look for the maximum value in this slice
            max_val = island_slice.max().item()
            
            # 2. Verify NO other higher marker is leaking in 
            # (If marker 1000 leaked into a marker 10 island, max_val would be ~1000)
            # We check if max_val is roughly equal to our expected marker
            is_isolated = (max_val > expected_marker * 0.1) and (max_val < expected_marker * 9.0)
            
            status = "[✓]" if is_isolated else "[X]"
            print(f"{status} Pathway: {p_name}")
            print(f"    - Expected Marker: {expected_marker} | Max Val Found: {max_val:.2f}")
            
            if not is_isolated:
                overall_success = False
                print(f"    - ERROR: Signal strength mismatch. Possible cross-talk or indexing drift.")
        
        current_ptr += num_nodes

    print("-"*70)
    if overall_success:
        print("  FINAL RESULT: ALL PATHWAYS ISOLATED SUCCESSFULLY")
    else:
        print("  FINAL RESULT: FAIL - CROSS-TALK DETECTED")
    print("="*70 + "\n")

def verify_gnn_stress_test1(g, model_layer, num_test_pathways=5):
    
    device = next(model_layer.parameters()).device

    subgraph_list = g.graph_data['batched_ppi_graph']

    print(f"Type of subgraph_list: {type(subgraph_list)}")
    if hasattr(subgraph_list, 'batch_size'):
        print(f"Is it a batch? {subgraph_list.batch_size > 1}")
    print(f"------------------------")

    if isinstance(subgraph_list, dgl.DGLGraph):
        print(f"Unbatching {subgraph_list.batch_size} pathways...")
        subgraph_list = dgl.unbatch(subgraph_list)
   
    total_genes = 0
    isolated_genes = 0
    pathways_with_isolation = 0

    for sg in subgraph_list:
        # Get in-degrees for the 'ppi' edge type
        # For a gene to be "connected," it needs at least one neighbor
        in_degrees = sg.in_degrees(etype='ppi')

        num_isolated = (in_degrees == 0).sum().item()

        total_genes += sg.num_nodes()
        isolated_genes += num_isolated

        if num_isolated > 0:
            pathways_with_isolation += 1

    print("--- ISOLATION AUDIT RESULTS ---")
    print(f"Total Pathways Scanned: {len(subgraph_list)}")
    print(f"Total Gene Instances:   {total_genes}")
    print(f"Isolated Genes (In-Degree 0): {isolated_genes}")
    print(f"Pathways with at least one isolated gene: {pathways_with_isolation}")

    if total_genes > 0:
        percent = (isolated_genes / total_genes) * 100
        print(f"Data Loss Risk without Self-Loops: {percent:.2f}%")
    print("-------------------------------")


    processed_subgraphs = []
    for sg in subgraph_list:
        # Use the first available edge type (likely 'ppi')
        etype = sg.etypes[0] 
        processed_subgraphs.append(dgl.add_self_loop(sg, etype=etype))
    
    #batched_g = dgl.batch(processed_subgraphs).to(device)
    batched_g = dgl.batch(subgraph_list).to(device)

    #batched_g = g.graph_data['batched_ppi_graph'].to(device)
    pathway_indices = g.graph_data['pathway_indices']

    nodes_per_p = batched_g.batch_num_nodes()
    global_nids = batched_g.ndata[dgl.NID].to(device)
    
    id2pathway = g.graph_data['id2pathway']
    id2gene = g.graph_data['id2gene']
    
    all_p_indices = list(range(len(pathway_indices)))
    sampled_p_indices = random.sample(all_p_indices, min(num_test_pathways, len(all_p_indices)))
    # 1. Clear everything
    test_h = torch.zeros((batched_g.num_nodes(), 128)).to(device)

    # 2. Inject with Audit
    batch_num_nodes = batched_g.batch_num_nodes()
    print(f"DEBUG: Batch contains {len(batch_num_nodes)} graphs.")

    cum_nodes = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch_num_nodes, dim=0)])

    print("\n--- INJECTION AUDIT ---")
    for i, p_list_idx in enumerate(sampled_p_indices):
        marker = 1.0 + (i * 2.0)
        start_idx = cum_nodes[p_list_idx].item()

        # Identify the gene we are actually touching in memory
        actual_node_id = batched_g.ndata[dgl.NID][start_idx].item()
        gene_name = id2gene[actual_node_id]

        p_name = id2pathway[pathway_indices[p_list_idx].item()]
        print(f"Targeting Pathway {p_name} | Memory Index: {start_idx} | Gene: {gene_name} | Value: {marker}")

        test_h[start_idx, :] = marker
        # ... (store infection_targets)
    print("-----------------------\n")

    # 3. Run Model
    out_h = model_layer(batched_g, test_h)
    if hasattr(out_h, 'squeeze'): out_h = out_h.squeeze(1)

    # 4. Verify with the SAME logic
    current_ptr = 0
    for i in range(len(batch_num_nodes)):
        num_nodes = batch_num_nodes[i].item()
        island_slice = out_h[current_ptr : current_ptr + num_nodes]

        if i in sampled_p_indices:
            # Check if the slice we are looking at has ANY signal
            slice_max = island_slice.max().item()
            i_name = id2pathway[pathway_indices[i].item()]
            print(f"Verifying Slice {i_name} (Indices {current_ptr}:{current_ptr+num_nodes}) | Max: {slice_max}")
            #audit_pathway_connectivity(g, i_name)

        current_ptr += num_nodes

def audit_pathway_connectivity(g, pathway_id_str):
    """
    Diagnostic: Checks if a pathway subgraph is actually 'navigable' by a GNN.
    """
    # 1. Find the internal integer ID for the Reactome String
    id2pathway = g.graph_data['id2pathway']
    pathway_dict = {v: k for k, v in id2pathway.items()}
    
    if pathway_id_str not in pathway_dict:
        print(f"Error: {pathway_id_str} not found in graph.")
        return

    p_idx = pathway_dict[pathway_id_str]
    
    # 2. Extract the subgraph for this specific pathway
    # This matches the logic used in your GNN batching
    src_p, dst_g = g.edges(etype='has_gene')

    mask = (src_p == p_idx)

    sub_nodes = dst_g[mask].unique()

    if len(sub_nodes) == 0:
        print(f"Error: No genes found for pathway {pathway_id_str}")
        return

    subp = dgl.node_subgraph(g, {'gene': sub_nodes})
    
    # 3. Analyze Edges
    num_nodes = subp.num_nodes('gene')
    num_edges = subp.num_edges('ppi') # Total edges in the PPI for this pathway
    
    # Calculate Degrees
    degrees = subp.in_degrees(etype='ppi')
    isolated_count = (degrees == 0).sum().item()
    
    print(f"\nAUDIT REPORT: {pathway_id_str}")
    print("="*40 + "\n")
    print(f"Total Genes in Pathway: {num_nodes}")
    print(f"Total PPI Edges Found:  {num_edges}")
    print(f"Isolated Nodes (0 edge): {isolated_count}")
    print(f"Graph Density:          {num_edges / (num_nodes * (num_nodes-1)) if num_nodes > 1 else 0:.4f}")
    
    if num_edges == 0:
        print(f"!! WARNING: This pathway is a 'Dead Zone'. No messages can pass.")

def verify_pathway_danger_zone(g):
    # 1. Check the size of the feature matrix
    total_pathway_rows = g.nodes['pathway'].data['feat'].shape[0]

    # 2. Check the IDs being used for the readout
    leaf_ids = g.graph_data['pathway_indices']
    max_leaf_id = leaf_ids.max().item()
    min_leaf_id = leaf_ids.min().item()

    print(f"Master Matrix Rows: {total_pathway_rows}")
    print(f"Leaf IDs range from: {min_leaf_id} to {max_leaf_id}")

    if max_leaf_id >= total_pathway_rows:
        print("❌ CRITICAL ERROR: Leaf IDs point outside the Master Matrix!")
    elif max_leaf_id < 1029 and total_pathway_rows > 1029:
        print("⚠️ WARNING: Your IDs are 'Local' (0-1028), but your Matrix is 'Global'. This is a Mismatch!")
    else:
        print("✅ SUCCESS: IDs and Matrix dimensions are aligned.")


def load_geo_context_metadata(meta_file):
    
    df = pd.read_csv(meta_file)
    
    meta = {}

    for _, row in df.iterrows():

        geo_id = row["geo_id"]

        meta[geo_id] = {
            "organ": row["organ"],
            "cell_type": row["cell_type"],
            "disease": row["disease"],
            "stimulus": row.get("stimulus", "unknown")
        }

    return meta

def load_geo_enrichment(enrichment_file, g, pathway2id, gene2id, padj_cutoff=0.05):

    geo_id = os.path.basename(enrichment_file).split("_")[0]

    #print(f"now processing file {geo_id}")

    pathway_genes = defaultdict(set)

    with open(enrichment_file) as f:
        
        reader = csv.DictReader(f)

        for row in reader:
            padj = float(row["p.adjust"])

            if padj > padj_cutoff:
                continue
            
            parent_pathway_name = row["ID"]

            if parent_pathway_name not in pathway2id:
                continue

            parent_pathway = pathway2id[parent_pathway_name]

            # -------------------
            # read enriched genes
            # -------------------
            gene_symbols = row["gene_symbols"].split(",")

            genes = set()

            for gsym in gene_symbols:
                gsym = gsym.strip()
                if gsym in gene2id:
                    gid = gene2id[gsym]
                    genes.add(gid)
                    #enriched_genes.add(gid)

            if not genes:
                continue

            # ------------------------
            # project to TAKE pathways
            # ------------------------
            leaf_pathways = project_to_take_pathways(g, parent_pathway, genes)

            for pw, overlap in leaf_pathways:
                if pw not in pathway_genes:
                    pathway_genes[pw] = {
                        "parent": parent_pathway_name,
                        "genes": set()
                    }

                pathway_genes[pw]["genes"].update(overlap)

    if not pathway_genes:
        return None

    ctx = GEOContext(
        geo_id = geo_id,
        pathway_genes = dict(pathway_genes)
    )

    return ctx

def load_all_geo_contexts(enrichment_dir, g, pathway2id, gene2id, padj_cutoff=0.05, top_k=None):

    contexts = []

    files = glob.glob(f"{enrichment_dir}/*_Reactome_enrichment_full.csv")

    for f in files:
        ctx = load_geo_enrichment(f, g, pathway2id, gene2id, padj_cutoff=padj_cutoff)

        if len(ctx.pathway_genes) == 0:
            continue

        contexts.append(ctx)

    return contexts

def project_to_take_pathways(g, parent_pathway, genes, ratio_threshold=0.6):

    leaf_cache = g.graph_data['leaf_descendants']

    leaf_pathways = leaf_cache.get(parent_pathway, [])

    if not leaf_pathways:
        leaf_pathways = [parent_pathway]

    scores = []

    for pw in leaf_pathways:
        # gene in this pathway
        pw_genes = g.graph_data['pathway_genes'][pw]

        if not pw_genes:
            continue

        pw_genes = set(pw_genes)
        overlap = genes & pw_genes

        if not overlap:
            continue

        overlap_ratio = len(overlap) / len(pw_genes)

        dice = 2 * len(overlap) / (len(genes) + len(pw_genes))

        geom = len(overlap) / (len(genes) * len(pw_genes)) ** 0.5

        jaccard = overlap_ratio * math.log1p(len(overlap))

        #scores.append((pw, overlap_ratio, overlap))
        scores.append((pw, geom, overlap))

    if not scores:
        return []

    scores.sort(key=lambda x: x[1], reverse=True)

    # Debug for monitoring scores
    parent_name = g.graph_data['id2pathway'][parent_pathway]
    #print(f"\n[Pathway projection debug]")
    #print(f"Parent pathway: {parent_name} (id={parent_pathway})")
    #print("Candidate leaf pathways:")

    best_score = scores[0][1]

    selected = []

    for pw, score, overlap in scores:
        pw_name = g.graph_data['id2pathway'][pw]
        #print(
        #    f"    child={pw_name} (id={pw}) "
        #    f"score={score:.3f} "
        #    f"overlap={len(overlap)}"
        #)
        if score >= best_score * ratio_threshold:
            selected.append((pw, overlap))
            #print(f" take this pathway")
        else:
            break
            #print(f" not to take this pathway")

    return selected

def build_context_id_maps(contexts):
    organs = sorted(set(ctx.organ for ctx in contexts))
    cells = sorted(set(ctx.cell_type for ctx in contexts))
    diseases = sorted(set(ctx.disease for ctx in contexts))
    stimuli = sorted(set(ctx.stimulus for ctx in contexts))

    organ2id = {o: i for i, o in enumerate(organs)}
    celltype2id = {c: i for i, c in enumerate(cells)}
    disease2id = {d: i for i, d in enumerate(diseases)}
    stimulus2id = {s: i for i, s in enumerate(stimuli)}

    return organ2id, celltype2id, disease2id, stimulus2id

def assign_context_ids(contexts, organ2id, celltype2id, disease2id, stimulus2id):

    for ctx in contexts:

        ctx.organ_id = organ2id[ctx.organ]
        ctx.cell_type_id = celltype2id[ctx.cell_type]
        ctx.disease_id = disease2id[ctx.disease]
        ctx.stimulus_id = stimulus2id[ctx.stimulus]

def sampler_sanity_test(sampler, ctx):

    samples = sampler.sample_pos_neg(ctx)

    gg_pos = samples["gg_pos"]
    gg_neg = samples["gg_neg"]
    gp_pos = samples["gp_pos"]
    gp_neg = samples["gp_neg"]

    print("\n===== SAMPLER SANITY TEST =====")

    print("GG positive:", len(gg_pos))
    print("GG negative:", len(gg_neg))

    print("GP positive:", len(gp_pos))
    print("GP negative:", len(gp_neg))

    # --------------------------------------------------
    # Check PPI consistency
    # --------------------------------------------------
    pos_not_ppi = 0
    pos_in_ppi = 0
    for g1, g2 in gg_pos:
        if (g1, g2) not in sampler.ppi_edges:
            pos_not_ppi += 1
        elif (g1, g2) in sampler.ppi_edges:
            pos_in_ppi += 1

    neg_are_ppi = 0
    for g1, g2 in gg_neg:
        if (g1, g2) in sampler.ppi_edges:
            neg_are_ppi += 1

    print("\nPPI check")
    print("GG positives NOT in PPI:", pos_not_ppi)
    print("GG positives in PPI:", pos_in_ppi)
    print("GG negatives ARE in PPI:", neg_are_ppi)

    # --------------------------------------------------
    # Pathway overlap check
    # --------------------------------------------------
    overlap_neg = 0
    for g1, g2 in gg_neg:
        p1 = sampler.gene2direct_pathways.get(g1, set())
        p2 = sampler.gene2direct_pathways.get(g2, set())

        if len(p1.intersection(p2)) > 0:
            overlap_neg += 1

    print("\nPathway overlap check")
    print("GG negatives sharing pathway:", overlap_neg)

    # --------------------------------------------------
    # Degree bias check
    # --------------------------------------------------
    pos_deg = []
    neg_deg = []

    for g1, g2 in gg_pos:
        pos_deg.append(sampler.gene_degree.get(g1,0))
        pos_deg.append(sampler.gene_degree.get(g2,0))

    for g1, g2 in gg_neg:
        neg_deg.append(sampler.gene_degree.get(g1,0))
        neg_deg.append(sampler.gene_degree.get(g2,0))

    print("\nDegree distribution")
    print("Pos mean degree:", np.mean(pos_deg))
    print("Neg mean degree:", np.mean(neg_deg))

    # --------------------------------------------------
    # Gene diversity check
    # --------------------------------------------------
    genes_pos = set()
    genes_neg = set()

    for g1, g2 in gg_pos:
        genes_pos.add(g1)
        genes_pos.add(g2)

    for g1, g2 in gg_neg:
        genes_neg.add(g1)
        genes_neg.add(g2)

    print("\nGene diversity")
    print("Unique genes in GG pos:", len(genes_pos))
    print("Unique genes in GG neg:", len(genes_neg))

    print("===== END TEST =====\n")

