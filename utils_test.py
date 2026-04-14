#!/usr/bin/env python3

import utils
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import dgl.nn.pytorch as dglnn
from collections import defaultdict

def main():
    print("Loading Reactome multiplex graph...\n")

    # Path to the required files
    ppi_file = 'data/merged_signaling_network_unique.tsv'
    pathway_relations_file = 'data/ReactomePathwaysRelation.MMU.gene.txt'
    pathway_gene_file = 'data/reactome_pathway_gene.csv'

    g, gene2id, pathway2id = utils.load_reactome_mux_graph(
            ppi_file,
            pathway_gene_file,
            pathway_relations_file)

    gene2pathway_dist = g.graph_data['gene2pathway_dist'] # for each gene, the distance from it to all pathways (through gene-gene interaction)
    id2gene = {v: k for k, v in gene2id.items()}
    id2pathway = {v: k for k, v in pathway2id.items()}

    utils.verify_batch_alignment(g)

    #utils.trace_gene_local_hierarchy('F2rl1', g)
    utils.verify_precomputed_hierarchy('F2rl1', g)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_layer = dglnn.GATConv(128,128,num_heads=1,feat_drop=0.0,attn_drop=0.0,allow_zero_in_degree=True).to(device)
    #model_layer = dglnn.GraphConv(128,128,norm='none',weight=True,bias=True).to(device)
    #utils.verify_gnn_isolation(g, model_layer)
    #utils.verify_gnn_stress_test(g, model_layer)
    #utils.verify_pathway_danger_zone(g)

    #utils.audit_pathway_connectivity(g, "R-MMU-203927")
    #utils.audit_pathway_connectivity(g, "R-MMU-9013148")


if __name__ == "__main__":
    main()

