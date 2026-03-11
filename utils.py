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

# class for GEO enrichment analysis file
class GEOContext:
    def __init__(self, geo_id, pathway_genes):
        self.geo_id = geo_id                  # GEO id
        self.pathway_genes = pathway_genes    # TAKE pathway ids

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

    # store pathway's chidren
    g.graph_data['leaf_descendants'] = build_leaf_descendant_cache(g)

    # store pathway genes
    g.graph_data['pathway_genes'] = {
        pw: set(g.successors(pw, etype='has_gene').tolist())
        for pw in range(g.num_nodes('pathway'))
    }

    # add the two id maps
    g.graph_data['id2pathway'] = id2pathway
    g.graph_data['id2gene'] = id2gene

    # Return the graph and the gene/pathway ID mappings
    return g, gene2id, pathway2id

def load_geo_enrichment(enrichment_file, g, pathway2id, gene2id, padj_cutoff=0.05):

    geo_id = os.path.basename(enrichment_file).split("_")[0]

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
    #parent_name = g.graph_data['id2pathway'][parent_pathway]
    #print(f"\n[Pathway projection debug]")
    #print(f"Parent pathway: {parent_name} (id={parent_pathway})")
    #print("Candidate leaf pathways:")

    best_score = scores[0][1]

    selected = []

    for pw, score, overlap in scores:
        #pw_name = g.graph_data['id2pathway'][pw]
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


