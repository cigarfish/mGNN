# test_gene_encoder.py
import torch
import dgl
from mhh_gnn import GeneEncoder, MHHGNN  # adjust import path if needed

# ---- Dummy PPI graph ----
num_genes = 10
edges_src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
edges_dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
g = dgl.graph((edges_src, edges_dst), num_nodes=num_genes)

# ---- Dummy features ----
in_dim = 16
feat = torch.randn(num_genes, in_dim)

# ---- Dummy pathway_nodes and gene2pathways maps ----
pathway_nodes = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8, 9],
}

gene2pathways = {
    0: [0],
    1: [0],
    2: [0],
    3: [1],
    4: [1],
    5: [1],
    6: [2],
    7: [2],
    8: [2],
    9: [2],
}

# -----------------------------
# 4. Initialize MHHGNN
# -----------------------------
gnn_type = "gcn"
num_layers = 2
hidden_dim = 16
out_dim = 12
attn_dim = 4
dropout_prob = 0.1

model = MHHGNN(
    gnn_type=gnn_type,
    num_layers=num_layers,
    in_dim=in_dim,
    hidden_dim=hidden_dim,
    out_dim=out_dim,
    attn_dim=attn_dim,
    dropout_prob=dropout_prob
)

# -----------------------------
# 5. Forward pass
# -----------------------------
outputs = model(g, feat, pathway_nodes, gene2pathways)

# -----------------------------
# 6. Inspect outputs
# -----------------------------
print("Gene embeddings:", outputs["gene_emb"].shape)
print("Pathway embeddings:", outputs["pathway_emb"].shape)
print("Gene-Pathway embeddings:", outputs["gene_pathway_emb"].shape)
print("Number of pathway attentions:", len(outputs["pathway_attn"]))
print("Number of gene attentions:", len(outputs["gene_attn"]))
