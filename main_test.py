#!/usr/bin/env python3

import utils
import torch
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

    gene2pathway_dist = g.graph_data['gene2pathway_dist']
    id2gene = {v: k for k, v in gene2id.items()}
    id2pathway = {v: k for k, v in pathway2id.items()}

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

