import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv, GINConv


class GeneEncoder(nn.Module):
    def __init__(
        self,
        g,
        gene2id,
        pathway2id,
        gnn_type: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        fanouts=None
    ):
        super(GeneEncoder, self).__init__()
        self.g = g
        self.gene2id = gene2id
        self.pathway2id = pathway2id
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.fanouts = fanouts if fanouts is not None else [10] * num_layers

        # GNN layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            out_d = out_dim if i == num_layers - 1 else hidden_dim

            self.layers.append(
                PathwayGNNLayer(
                    gnn_type,
                    in_d,
                    out_d,
                    dropout=dropout
                )
            )

    def get_pathway_genes(self, pathway_id):
        """
        Get the list of genes associated with a given pathway
        """
        if isinstance(pathway_id, str):
            if pathway_id not in self.pathway2id:
                raise ValueError(f"Pathway {pathway_id} not found in pathway2id mapping")
            p_idx = self.pathway2id[pathway_id]
        else:
            p_idx = pathway_id

        gene_indices = self.g.successors(p_idx, etype='has_gene').tolist()

        return gene_indices

    def get_local_ppi_subgraph(self, pathway_id):
        """
        Get the local PPI subgraph
        """
        genes_in_pathway = self.get_pathway_genes(pathway_id)
        ppi_subgraph = self.g.subgraph(genes_in_pathway)
        return ppi_subgraph

    def get_leaf_pathways(self):
        """
        Extract all leaf pathways
        """
        return self.g.nodes['pathway'].data['take_mask'].nonzero(as_tuple=True)[0]

    def forward(self, pathway_id: int): #g: dgl.DGLHeteroGraph):
        """
        Input:
            g.nodes['gene'].data['feat']  -> (N_gene, in_dim)

        Output:
            gene_embeddings               -> (N_gene, out_dim)
        """
        device = next(self.parameters()).device

        # step 1: get seed genes in the given pathway
        seed_genes = self.get_pathway_genes(pathway_id)
        seed_genes = torch.tensor(seed_genes, device=device)

        # step 2: build k-hop neighbors
        # option 1: sample all neighboring nodes; option 2: do fixed fanouts
        #subgraph = self.get_local_ppi_subgraph(pathway_id).to(device)
        ppi_g = self.g['gene', 'ppi', 'gene']

        # Create a tensor of edge weights
        pathway_mask = self.g.edges['ppi'].data['pathway_mask'][:, pathway_id]
        edge_prob = torch.ones(ppi_g.num_edges(), device=device)
        edge_prob[pathway_mask] = 3.0

        # step 3: create blocks (k-hop neighborhoods)
        # neighbor sampler with pathway bias
        sampler = dgl.dataloading.MultiLayerNeighbor(self.fanouts)
        blocks = sampler.sample_blocks(ppi_g, seed_genes, prob=edge_prob)
        #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
        #blocks = sampler.sample_blocks(subgraph, seed_genes)

        # step 4: prepare input features
        h = blocks[0].srcdata['feat']
        #attentions = []

        # step 5: forward through GNN layers
        for (layer, block) in enumerate(zip(self.layers, blocks)):
            h, alpha = layer(block, h)
            #attentions.append(alpha)

        return h_out, alpha #attentions

class PathwayGNNLayer(nn.Module):
    def __init__(
        self,
        gnn_type: str,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        self.gnn_type = gnn_type.lower()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation) if activation else None

        # ---------- base GNN ----------
        if self.gnn_type == "gcn":
            self.gnn = GraphConv(
                in_dim, out_dim,
                norm="both",
                allow_zero_in_degree=True
            )
        elif self.gnn_type == "gin":
            self.gnn = GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                ),
                aggregator_type="sum"
            )
        elif self.gnn_type == "gat":
            self.gnn = GATConv(
                in_dim, out_dim,
                num_heads=1,
                feat_drop=dropout,
                allow_zero_in_degree=True
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # ---------- gene–pathway attention ----------
        self.gene_attn = nn.Linear(out_dim, 1, bias=False)

    def forward(self, block, h):
        """
        block: DGLBlock (k-hop neighborhood)
        h:     (num_src_nodes, in_dim)
        """

        # message passing
        if self.gnn_type in ["gcn", "gin"]:
            h = self.gnn(block, h)

        elif self.gnn_type == "gat":
            h = self.gnn(block, h).squeeze(1)

        # keep dst nodes only
        num_dst = block.number_of_dst_nodes()
        h = h[:num_dst]

        # gene importance
        alpha = torch.sigmoid(self.gene_attn(h))  # (num_src, 1)
        h = h * alpha

        return h, alpha



class PathwayEncoder(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, attn_dim, bias=False)
        self.a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, gene_embeds):
        """
        gene_embeds: (N_p, d)
        """
        scores = self.a(torch.tanh(self.W(gene_embeds)))  # (N_p, 1)
        alpha = torch.softmax(scores, dim=0)              # normalize over genes
        pathway_embed = torch.sum(alpha * gene_embeds, dim=0)
        return pathway_embed, alpha
