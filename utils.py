#!/usr/bin/env python3

from collections import defaultdict, deque
from functools import partial, reduce
import json
import math
import multiprocessing
import random

import dgl
import networkx as nx
import numpy as np
import torch
import pandas as pd

from dataset import GraphDataset

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

    ppi_edges_ids = []
    for _, row in df_ppi.iterrows():
        if row['gene1'] in gene2id and row['gene2'] in gene2id:
            ppi_edges_ids.append((gene2id[row['gene1']], gene2id[row['gene2']]))

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

    # Return the graph and the gene/pathway ID mappings
    return g, gene2id, pathway2id
