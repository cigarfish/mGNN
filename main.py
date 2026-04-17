#!/usr/bin/env python3

import utils
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from model.mhh_gnn import MHHGNN
from model.sample_test import PathwayNegativeSampler, AnalyticReplaySampler

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

    # loading enrichment files
    contexts = utils.load_all_geo_contexts(
        "data/enrichment",
        g,
        pathway2id,
        gene2id
    )
    # loading geo meta file
    meta = utils.load_geo_context_metadata("data/geo_context_metadata.csv")

    print(f"\nLoaded {len(contexts)} GEO enrichment contexts\n")
    
    for ctx in contexts:
        print(f"GEO dataset: {ctx.geo_id}")
        print(f"Number of enriched pathways: {len(ctx.pathway_genes)}")
        for pid, info in ctx.pathway_genes.items():
            pname = id2pathway.get(pid, "UNKNOWN")
            parent = info["parent"]
            genes = info["genes"]
            gene_symbols = [id2gene[g] for g in list(genes)[:10]] # show first 10

        info = meta.get(ctx.geo_id)

        if info is None:
            print(f"Warning: no metadata for ", ctx.geo_id)
            continue

        ctx.organ = info["organ"]
        ctx.cell_type = info["cell_type"]
        ctx.disease = info["disease"]
        ctx.stimulus = info["stimulus"]

        print("-" * 60)

    organ2id, celltype2id, disease2id, stimulus2id = utils.build_context_id_maps(contexts)

    utils.assign_context_ids(contexts, organ2id, celltype2id, disease2id, stimulus2id)

    context_info = {
        'organ_map': organ2id,
        'cell_map': celltype2id,
        'disease_map': disease2id,
        'stimulus_map': stimulus2id,
        'counts': {
            'organ': len(organ2id),
            'cell': len(celltype2id),
            'disease': len(disease2id),
            'stimulus': len(stimulus2id)
        }
    }

    # create sampler
    sampler = PathwayNegativeSampler(g)
    analyticReplaySampler = AnalyticReplaySampler(sampler)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create MHH_GNN
    model = MHHGNN(
        context_info=context_info,
        in_dim=128,
        hidden_dim=64,
        emb_dim=12,
        device=device
    )
    model = model.to(device)

    # training
    model.train()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20

    for epoch in range(num_epochs):
        total_loss = 0
        for ctx in contexts:
            optimizer.zero_grad()
            loss, _ = model(
                g, ctx, sampler=sampler, compute_loss=True
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: {total_loss:.4f}")


if __name__ == "__main__":
    main()

