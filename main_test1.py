#!/usr/bin/env python3

import utils
import torch
import numpy as np
from collections import defaultdict
from model.mhh_gnn import MHHGNN
from model.sample_test import PathwayNegativeSampler
from utils import sampler_sanity_test

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
            #print(
            #    f"    parent={parent} -> "
            #    f"pathway_id={pid} "
            #    f"pathway_name={pname} "
            #    f"num_genes={len(genes)} "
            #    f"sample_genes={gene_symbols}"
            #)

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

    print("\n==== CONTEXT DICTIONARIES ====")
    print("\nOrgan -> ID")
    for k, v in organ2id.items():
        print(f"{k}: {v}")

    print("\nCell type -> ID")
    for k, v in celltype2id.items():
        print(f"{k}: {v}")

    print("\nDisease -> ID")
    for k, v in disease2id.items():
        print(f"{k}: {v}")

    print("\nStimulus -> ID")
    for k, v in stimulus2id.items():
        print(f"{k}: {v}")

    # create sampler
    sampler = PathwayNegativeSampler(g)

    # Create MHH_GNN
    model = MHHGNN(
        num_organs=len(organ2id),
        num_cells=len(celltype2id),
        num_diseases=len(disease2id),
        num_stimuli=len(stimulus2id),
        in_dim=128,
        hidden_dim=64,
        emb_dim=12
    )
    # Debug graph in MHH_GNN
    model.debug_graph_consistency(g)
    for context_id, ctx in enumerate(contexts):
        print(f"\nRunning forward for context {context_id} ({ctx.geo_id}) | ")
        print(f"Organ: {ctx.organ} ({ctx.organ_id}) | ")
        print(f"Cell: {ctx.cell_type} ({ctx.cell_type_id}) | ")
        print(f"Disease: {ctx.disease} ({ctx.disease_id}) | ")
        print(f"Stimulus: {ctx.stimulus} ({ctx.stimulus_id})")
        outputs = model.forward(g, ctx)

        gene_emb = outputs['gene_emb']
        pathway_emb = outputs['pathway_emb']
        gene_pathway_emb = outputs['gene_pathway_emb']
        pathway_attn = outputs['pathway_attn']
        gene_attn = outputs['gene_attn']

        # Print shapes to check
        print("Gene embeddings:", gene_emb.shape)
        print("Pathway embeddings:", pathway_emb.shape)
        print("Gene-Pathway embeddings:", gene_pathway_emb.shape)

        if context_id == 0:
            emb0 = pathway_emb.detach()
            sampler_sanity_test(sampler, ctx)
        if context_id == 1:
            diff = (pathway_emb - emb0).abs().mean()
            print(f"Context embedding difference: ", diff.item())
            sampler_sanity_test(sampler, ctx)

    # -----------------------------
    # Basic graph info
    # -----------------------------
    print("\n=== Graph Summary ===")
    print(g)

    print("Node types:", g.ntypes)
    print("Edge types:", g.etypes)

    # -----------------------------
    # Node counts
    # -----------------------------
    print("\n=== Node Counts ===")
    for ntype in g.ntypes:
        print(f"{ntype}: {g.num_nodes(ntype)}")

    # -----------------------------
    # Edge counts
    # -----------------------------
    print("\n=== Edge Counts ===")
    for etype in g.canonical_etypes:
        print(f"{etype}: {g.num_edges(etype)}")

    # -----------------------------
    # Mapping sanity checks
    # -----------------------------
    print("\n=== Mapping Checks ===")
    print(f"Total genes in gene2id: {len(gene2id)}")
    print(f"Total pathways in pathway2id: {len(pathway2id)}")

    print("\n=== Feature Shapes ===")
    print(f"  Gene Feats:    {g.nodes['gene'].data['feat'].shape}")
    print(f"  Pathway Feats: {g.nodes['pathway'].data['feat'].shape}")

    # Show a few examples
    print("\nExample gene IDs:")
    for i, (gene, gid) in enumerate(gene2id.items()):
        print(f"  {gene} -> {gid}")
        if i >= 4:
            break

    print("\nExample pathway IDs:")
    for i, (pid, pid_id) in enumerate(pathway2id.items()):
        print(f"  {pid} -> {pid_id}")
        if i >= 4:
            break

    # -----------------------------
    # Edge consistency checks
    # -----------------------------
    print("\n=== Consistency Checks ===")

    # gene -> pathway edges should match pathway -> gene
    gp_edges = g.num_edges(('gene', 'in_pathway', 'pathway'))
    pg_edges = g.num_edges(('pathway', 'has_gene', 'gene'))
    src, dst = g.edges(etype=('gene', 'in_pathway', 'pathway'))
    take_mask = g.nodes['pathway'].data['take_mask']
    valid_mask = take_mask[dst]
    src = src[valid_mask]
    dst = dst[valid_mask]
    gene_pathway_deg = torch.bincount(
            src,
            minlength=g.num_nodes('gene')
    ).cpu().numpy()
    print("Mean pathways per gene:", gene_pathway_deg.mean())
    print("Max pathways per gene:", gene_pathway_deg.max())
    bins = [0,1,2,3,4,5,6,7,8,9,10]
    hist = {}
    for b in bins:
        hist[b] = np.sum(gene_pathway_deg == b)
    hist["10+"] = np.sum(gene_pathway_deg > 10)
    print("\nGene -> pathway count histogram")
    for k,v in hist.items():
        print(f"{k}: {v}")

    print(f"gene→pathway edges: {gp_edges}")
    print(f"pathway→gene edges: {pg_edges}")
    assert gp_edges == pg_edges, "Mismatch in gene-pathway edge counts!"

    # parent-child pathway edges should be paired
    pc = g.num_edges(('pathway', 'parent_of', 'pathway'))
    cp = g.num_edges(('pathway', 'child_of', 'pathway'))
    print(f"parent_of edges: {pc}")
    print(f"child_of edges:  {cp}")
    assert pc == cp, "Mismatch in pathway hierarchy edges!"

    print("\n=== Mask Checks ===")
    if 'take_mask' in g.nodes['pathway'].data:
        mask = g.nodes['pathway'].data['take_mask']
        num_take_1 = torch.sum(mask).item()
        num_take_0 = mask.shape[0] - num_take_1

        print(f" Pathways with TAKE = 1 (Leaf): {num_take_1}")
        print(f" Pathways with TAKE = 0 (Leaf): {num_take_0}")

        test_pathway = 'R-MMU-3215018'
        if test_pathway in pathway2id:
            pidx = pathway2id[test_pathway]
            is_leaf = mask[pidx].item()
            print(f" Manual Check [{test_pathway}]: {'LEAF (TAKE=1)' if is_leaf else 'ROOT (TAKE=0)'}")

    take_mask = g.nodes['pathway'].data['take_mask']
    target_pathway = 'R-MMU-3371378' #'R-MMU-3215018'
    if target_pathway in pathway2id:
        p_idx = pathway2id[target_pathway]

        gene_indices = g.successors(p_idx, etype='has_gene').tolist()

        id2gene = {v: k for k, v in gene2id.items()}

        print(f"Analyzing pathway: {target_pathway} (Internal index: {p_idx})")
        print(f"{'Mapped ID':<12} | {'Gene Name'}")
        for g_idx in gene_indices[:15]:
            gene_name = id2gene[g_idx]
            print(f"{g_idx:<12} | {gene_name}")

        if len(gene_indices) > 15:
            print(f"... and {len(gene_indices) - 15} more genes.\n")


        # Debug: check specific edge
        #print("Rwdd2b in gene2id:", "Rwdd2b" in gene2id)
        #print("Ube2i in gene2id:", "Ube2i" in gene2id)

        #print("RWDD2B in gene2id:", "RWDD2B" in gene2id)
        #print("UBE2I in gene2id:", "UBE2I" in gene2id)

        # test edge-pathway mask behavior
        ppi_g = g['gene', 'ppi', 'gene']
        pathway_mask = g.edges['ppi'].data['pathway_mask']

        src, dst = ppi_g.edges()
        num_edges = ppi_g.num_edges()

        print("\n========== PPI–Pathway Mask Test ==========")
        print(f"Target pathway: {target_pathway}  (p_idx = {p_idx})")
        print(f"Total PPI edges: {num_edges}")
        print("==========================================")

        gene_set = set(gene_indices)

        inside_edges = []
        outside_edges = []

        for e in range(num_edges):
            u = src[e].item()
            v = dst[e].item()

            in_pathway_edge = (u in gene_set and v in gene_set)
            mask_val = pathway_mask[e, p_idx].item()

            if in_pathway_edge:
                inside_edges.append((u, v, mask_val))
            else:
                outside_edges.append((u, v, mask_val))

        id2gene = {v: k for k, v in gene2id.items()}

        print("\n--- Edges where BOTH genes are in pathway ---")
        for u, v, m in inside_edges[:20]:
            print(f"{id2gene[u]} -- {id2gene[v]} | mask[{target_pathway}] = {bool(m)}")

        print(f"... total in-pathway PPIs: {len(inside_edges)}")

        print("\n--- Edges where genes are NOT both in pathway ---")
        for u, v, m in outside_edges[:20]:
            print(f"{id2gene[u]} -- {id2gene[v]} | mask[{target_pathway}] = {bool(m)}")

        print(f"... total non-pathway PPIs: {len(outside_edges)}")

        # ===== sanity checks =====
        if len(inside_edges) > 0:
            true_ratio = sum(int(m) for _, _, m in inside_edges) / len(inside_edges)
            print(f"\nSanity check: in-pathway edge mask true ratio = {true_ratio:.4f}")

        if len(outside_edges) > 0:
            false_ratio = sum(1 - int(m) for _, _, m in outside_edges) / len(outside_edges)
            print(f"Sanity check: out-pathway edge mask false ratio = {false_ratio:.4f}")

        # 1 genes directly in pathway (distance should be 0)
        genes_in_pathway = g.successors(p_idx, etype='has_gene').tolist()

        print("--- Direct members (expect distance = 0) ---")
        unique_pathways = set()
        for g_idx in genes_in_pathway[:10]:
            dist = gene2pathway_dist.get(g_idx, {}).get(p_idx, None)
            pathways = {p for p,d in gene2pathway_dist[g_idx].items() if d == 0}
            unique_pathways.update(pathways)
            print(
                f"{id2gene[g_idx]:<12} -> dist = {dist} | "
                f"#direct pathways = {len(pathways)}"
                )

        print(f"Total unique direct pathways across genes: {len(unique_pathways)}")

        # 2 genes NOT in pathway but reachable
        print("\n--- Reachable but NOT direct members (dist > 0) ---")
        cnt = 0
        for g_idx, p_map in gene2pathway_dist.items():
            if p_idx in p_map and g_idx not in genes_in_pathway:
                print(f"{id2gene[g_idx]:<12} -> dist = {p_map[p_idx]}")
                cnt += 1
            if cnt == 10:
                break

        if cnt == 0:
            print("No indirect genes found (check max_hops or data).")

        # 3 unreachable genes
        print("\n--- Unreachable genes (should NOT appear) ---")
        for g_idx in list(gene2id.values())[:10]:
            if p_idx not in gene2pathway_dist.get(g_idx, {}):
                print(f"{id2gene[g_idx]:<12} -> unreachable")
                
        rows = []

        for g_idx, dist_map in gene2pathway_dist.items():
            if p_idx not in dist_map:
                continue

            dist = dist_map[p_idx]
            gene_name = id2gene[g_idx]

            # pathways this gene directly belongs to
            direct_pathways = g.successors(g_idx, etype='in_pathway').tolist()
            pathway_names = [id2pathway[p] for p in direct_pathways]

            rows.append((dist, gene_name, pathway_names))

        # sort by distance, then gene name
        rows.sort(key=lambda x: (x[0], x[1]))

        #for dist, gene_name, pathway_names in rows:
        #    print(f"{gene_name:<12} | dist = {dist:<2} | pathways:")
        #    for pname in pathway_names:
        #        print(f"    - {pname}")
        #    print()

        genes_per_dist = defaultdict(int)
        pathways_per_dist = defaultdict(set)
        all_pathways = set()

        for g_idx, dist_map in gene2pathway_dist.items():
            if p_idx not in dist_map:
                continue

            dist = dist_map[p_idx]
            genes_per_dist[dist] += 1

            direct_pathways = g.successors(g_idx, etype='in_pathway').tolist()
            for pw in direct_pathways:
                if not take_mask[pw]:
                    continue
                pathways_per_dist[dist].add(pw)
                all_pathways.add(pw)

        print("\n========== Reachability Summary ==========")
        print(f"Target pathway: {target_pathway}\n")

        for dist in sorted(genes_per_dist):
            pids = sorted(pathways_per_dist[dist])
            names = [id2pathway[p] for p in pids[:20]]
            print(
                f"dist = {dist:<2} | "
                f"#genes = {genes_per_dist[dist]:<4} | "
                f"#pathways = {len(pathways_per_dist[dist])} | "
                f"examples: {', '.join(names)}" +
                (f" ... (+{len(pids) - 20} more)" if len(pids) > 20 else "")
            )

        print("\n----------------------------------------")
        print(f"Total reachable genes    : {sum(genes_per_dist.values())}")
        print(f"Total unique pathways    : {len(all_pathways)}")
        print("========================================")

    print("\n✅ Graph sanity check passed.")


if __name__ == "__main__":
    main()

