# test_gene_encoder.py
import torch
import dgl
import numpy as np
from mhh_gnn import GeneEncoder, PathwayAttentionLayer

def test_case_attention():
    hidden_dim = 16

    # 1. Define the 8 pathways from your F2rl1 data
    # Real Leaf (0), Local Leaf (1), Parent (2)
    pathway_names = [
        "R-MMU-749448", "R-MMU-375276", "R-MMU-416476", "R-MMU-373076",
        "R-MMU-500792", "R-MMU-388396", "R-MMU-372790", "R-MMU-162582"
    ]
    leaf_types = torch.tensor([0, 1, 2, 2, 2, 2, 2, 2]) # Types from your table
    depths = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])     # Depths from your table

    # 2. Build Graph (8 pathways -> 1 gene)
    u = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]) # pathway indices
    v = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]) # all point to gene 0
    g = dgl.heterograph({('pathway', 'has_gene', 'gene'): (u, v)})

    # 3. Assign Metadata to edges
    etype = ('pathway', 'has_gene', 'gene')
    g.edges[etype].data['rel_depth'] = depths
    g.edges[etype].data['leaf_type'] = leaf_types

    # 4. Initialize dummy embeddings
    # To test ONLY the hierarchy impact, we set embeddings to a constant
    # so the raw_score is 0 for everyone. This isolates the bias effect.
    pathway_emb = torch.ones(8, hidden_dim)
    gene_emb = torch.ones(1, hidden_dim)
    ctx_vec = torch.ones(1, hidden_dim)

    # 5. Initialize Layer
    model = PathwayAttentionLayer(hidden_dim, max_rel_depth=15)

    # 6. Run Forward Pass
    _ = model(g, pathway_emb, gene_emb, ctx_vec)

    # 7. Analyze Alpha Weights
    alphas = g.edges[etype].data['a'].detach().numpy()

    print(f"{'PATHWAY':<15} | {'TYPE':<10} | {'DEPTH':<5} | {'ATTENTION %'}")
    print("-" * 55)
    for i, name in enumerate(pathway_names):
        t_str = ["Real", "Local", "Parent"][leaf_types[i]]
        print(f"{name:<15} | {t_str:<10} | {depths[i]:<5} | {alphas[i]*100:>10.2f}%")

def debug_segment_max():
    print("=== Testing Segment Max Logic ===")
    
    # 1. Setup Dummy Data
    # 5 unique genes, 10 total instances
    num_genes = 5
    num_instances = 10
    device = torch.device('cpu')
    
    # Randomly assign instances to Gene IDs (0 to 4)
    global_nids = torch.tensor([0, 1, 0, 2, 1, 0, 3, 4, 2, 0], dtype=torch.long)
    
    # Create random scores
    instance_scores = torch.randn(num_instances, 1) * 10  # Scale up to test range
    
    # 2. Run the "Fast" Vectorized Logic
    gene_max = torch.full((num_genes, 1), -1e9, device=device)
    # Using index_reduce_ (PyTorch 1.12+)
    gene_max.index_reduce_(0, global_nids, instance_scores, reduce='amax', include_self=False)
    
    # 3. Run the "Slow" Ground Truth Logic (Verification)
    ground_truth_max = torch.full((num_genes, 1), -1e9, device=device)
    for i in range(num_instances):
        gene_id = global_nids[i].item()
        score = instance_scores[i].item()
        if score > ground_truth_max[gene_id]:
            ground_truth_max[gene_id] = score
            
    # 4. Compare
    diff = torch.abs(gene_max - ground_truth_max).max().item()
    
    # 5. Print Results
    print(f"Instance Scores:\n{instance_scores.flatten()}")
    print(f"Global NIDs:     {global_nids.tolist()}")
    print("-" * 30)
    print(f"Vectorized Max:  {gene_max.flatten()}")
    print(f"Ground Truth:    {ground_truth_max.flatten()}")
    
    if diff < 1e-6:
        print("\n✅ SUCCESS: Vectorized Max matches Ground Truth!")
    else:
        print(f"\n❌ FAILURE: Max difference is {diff}")


if __name__ == "__main__":
    #test_case_attention()
    debug_segment_max()

