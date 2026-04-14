import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn import GraphConv, GATConv, GINConv

from model.loss import PathwayNegativeSamplingLoss

class GeneEncoder(nn.Module):
    """
    Gene-level GNN encoder using local PPI subgraphs. Supporting GCN, GAT, and GIN
    """
    def __init__(
        self,
        gnn_type: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        pathway_feat_dim: int, # create before model creation: g.nodes['pathway'].data['feat'].shape[1]
        attn_dim: int = 16,
        num_layers: int = 2,
        dropout_prob = 0.2,
        mode="baseline"
    ):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.pathway_feat_dim = pathway_feat_dim
        self.gene_pooling = "mean" # mean or size_weighted
        # dimension projection
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.input_norm = nn.LayerNorm(out_dim)
        # mode
        self.mode = mode # baseline / unified / mux / hybrid
        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = out_dim if i == 0 else hidden_dim
            layer_out = out_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(self.build_layer(layer_in, layer_out))

        self.dropout_layer = nn.Dropout(self.dropout_prob)
        # Gene-pathway attention
        self.gene_attn_W = nn.Linear(out_dim * 2, attn_dim, bias=False)
        self.gene_attn_a = nn.Linear(attn_dim, 1, bias=False)
        self.pathway_proj = nn.Linear(pathway_feat_dim, out_dim)
        # Pathway-gene attention
        self.pathway_attn_W = nn.Linear(out_dim * 2, attn_dim, bias=False)
        self.pathway_attn_a = nn.Linear(attn_dim, 1, bias=False)
        # leaf, parent logit
        self.leaf_logit = nn.Parameter(torch.tensor(0.0)) # exp(0) = 1
        self.parent_logit = nn.Parameter(torch.tensor(-0.7)) # exp(0.7) = 0.5
        # context, hierarchy scaling
        self.context_lambda = nn.Parameter(torch.tensor(1.0))
        self.struct_lambda = nn.Parameter(torch.tensor(1.0))
        # Unified attention
        self.attn_W = nn.Linear(out_dim * 3, attn_dim, bias=False)
        self.attn_a = nn.Linear(attn_dim, 1, bias=False)
        self.readout_W = nn.Linear(out_dim, attn_dim, bias=False)
        self.readout_a = nn.Linear(attn_dim, 1, bias=False)
        # Mux attention
        self.mux_attn_W = nn.Linear(out_dim, attn_dim, bias=False)
        self.mux_attn_a = nn.Linear(attn_dim, 1, bias=False)

    def build_layer(self, in_dim, out_dim):
        if self.gnn_type == "gcn":
            return GraphConv(in_dim, out_dim, norm="both", allow_zero_in_degree=True)
        elif self.gnn_type == "gat":
            return GATConv(in_dim, out_dim, num_heads=1, feat_drop=0.2, allow_zero_in_degree=True)
        elif self.gnn_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINConv(mlp, aggregator_type="sum")
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

    def forward_baseline(self, g, feat, context_weights=None):
        """
        Input:
            g: DGLGraph (can be full PPI or k-hop subgraph)
        """
        device = feat.device
        h = self.input_proj(feat)
        h = self.input_norm(h)
        # 1. Message passing (PPI)
        num_pathways = g.num_nodes('pathway')
        edge_mask = g.edges(etype='ppi').data['pathway_mask']
        h_pathway = [] # list of (num_genes, dim)

        for p in range(num_pathways):
            # skip parent pathways
            if not g.nodes['pathway'].data['take_mask'][p]:
                continue
            # a. select edges belonging to pathway p
            eids = torch.nonzero(edge_mask[:,p], as_tuple=True)[0]
            if eids.numel() == 0:
                continue
            # b. build subgraph
            subg = dgl.edge_subgraph(g['ppi'], eids, preserve_nodes=True)
            h_p = h.clone()
            # c. run GNN on this pathway-specific graph
            for layer in self.layers:
                if self.gnn_type == "gat":
                    h_p = layer(subg, h_p).squeeze(1)
                else:
                    h_p = layer(subg, h_p)
                h_p = F.relu(h_p)
                h_p = self.dropout_layer(h_p)
            h_pathway.append((p, h_p))
        # 2. Gene -> Pathway attention
        pathway_feat = g.nodes['pathway'].data['feat']
        pathway_query = self.pathway_proj(pathway_feat)
        num_pathways = g.num_nodes('pathway')
        if len(h_pathway) == 0:
            dim = self.out_dim
        else:
            dim = h_pathway[0][1].size(1)
        pathway_emb = torch.zeros(num_pathways, dim, device=pathway_feat.device)
        g2p_src = [] # gene ids
        g2p_dst = [] # pathway ids
        g2p_attn = [] # attention weights
        for (p, h_p) in h_pathway:
            # a. get genes in this pathway
            gene_ids = g.predecessors(p, etype='in_pathway')
            if gene_ids.numel() == 0:
                continue
            # b. gather embeddings for these genes (pathway-specific)
            h_genes = h_p[gene_ids]
            # c. compute attention scores
            p_query = pathway_query[p]
            p_query_expand = p_query.unsqueeze(0).expand_as(h_genes)
            # d. joint attention
            attn_input = torch.cat([h_genes, p_query_expand], dim=1)
            h_proj = torch.tanh(self.gene_attn_W(attn_input))
            attn_scores = self.gene_attn_a(h_proj).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=0)
            g2p_src.append(gene_ids)
            g2p_dst.append(torch.full_like(gene_ids, p, device=gene_ids.device))
            g2p_attn.append(attn_weights)
            # e. aggregate
            p_emb = torch.sum(attn_weights.unsqueeze(-1) * h_genes, dim=0)
            # f. assign
            pathway_emb[p] = p_emb
        if len(g2p_attn) > 0:
            g2p_src = torch.cat(g2p_src)
            g2p_dst = torch.cat(g2p_dst)
            alpha_gp = torch.cat(g2p_attn)
        else:
            g2p_src = torch.tensor([], dtype=torch.long, device=pathway_feat.device)
            g2p_dst = torch.tensor([], dtype=torch.long, device=pathway_feat.device)
            alpha_gp = torch.tensor([], device=pathway_feat.device)
        # 3. Pathway -> Gene attention
        src_p, dst_g = g.edges(etype='has_gene')
        p_feat = pathway_emb[src_p]
        g_feat = h[dst_g]
        attn_input = torch.cat([p_feat, g_feat], dim=1)
        scores = self.pathway_attn_a(
            torch.tanh(self.pathway_attn_W(attn_input))
        ).squeeze(-1)
        # add context here
        if context_weights is not None:
            w_edge = context_weights[src_p]
            scores = scores + self.context_lambda * torch.log(w_edge + 1e-8)
        take_mask = g.nodes['pathway'].data['take_mask'][src_p]
        scores = scores + self.struct_lambda * torch.where(
            take_mask == 1,
            self.leaf_logit,
            self.parent_logit
        )
        alpha = dgl.ops.edge_softmax(g['has_gene'], scores)
        gene_emb = torch.zeros(
            g.num_nodes('gene'),
            p_feat.size(1),
            device=p_feat.device
        )
        gene_emb.index_add_(0, dst_g, alpha.unsqueeze(-1) * p_feat)
        return {
            "gene_emb": gene_emb,
            "pathway_emb": pathway_emb,
            "gene_hidden": h,
            "p2g_edge_index": (src_p, dst_g),
            "p2g_edge_attn": alpha,
            "g2p_edge_index": (g2p_src, g2p_dst),
            "g2p_edge_attn": alpha_gp
        }

    def forward_unified(self, g, feat, context_weights=None):
        # Unified Multiplex + Attention
        h = self.input_proj(feat)
        h = self.input_norm(h)
        pathway_feat = g.nodes['pathway'].data['feat']
        pathway_query = self.pathway_proj(pathway_feat)
        num_genes = g.num_nodes('gene')
        num_pathways = g.num_nodes('pathway')
        edge_mask = g.edges(etype='ppi').data['pathway_mask']
        gene_emb_per_pathway = []
        g2p_attn_records = []
        src, dst = g.edges(etype='ppi')
        for p in range(num_pathways):
            # skip parent pathway
            if not g.nodes['pathway'].data['take_mask'][p]:
                continue
            eids = torch.nonzero(edge_mask[:, p], as_tuple=True)[0]
            if eids.numel() == 0:
                continue
            subg = dgl.edge_subgraph(g['ppi'], eids, preserve_nodes=True)
            src_p, dst_p = subg.edges()
            # 1. get pathway query
            p_q = pathway_query[p]
            # 2. expand to edges
            p_q_expand = p_q.unsqueeze(0).expand(src_p.shape[0], -1)
            h_src = h[src_p]
            h_dst = h[dst_p]
            # 3. attention input
            attn_input = torch.cat([h_dst, h_src, p_q_expand], dim=1)
            h_proj = torch.tanh(self.attn_W(attn_input))
            scores = self.attn_a(h_proj).squeeze(-1)
            # 5. normalize per destination node
            alpha = dgl.ops.edge_softmax(subg, scores)
            # 6. aggregate
            h_p = torch.zeros_like(h)
            h_p.index_add_(0, dst_p, alpha.unsqueeze(-1) * h_src)
            h_p = F.relu(h_p)
            h_p = self.dropout_layer(h_p)
            gene_emb_per_pathway.append((p, h_p))
            # 7. store attention
            g2p_attn_records.append((src_p, dst_p, alpha, p))
        pathway_emb = torch.zeros(num_pathways, h.size(1), device=h.device)
        g2p_alpha_dict = {}
        for (p, h_p) in gene_emb_per_pathway:
            gene_ids = g.predecessors(p, etype='in_pathway')
            if gene_ids.numel() == 0:
                continue
            h_genes = h_p[gene_ids]
            scores = self.readout_a(torch.tanh(self.readout_W(h_genes))).squeeze(-1)
            alpha = torch.softmax(scores, dim=0)
            p_emb = torch.sum(alpha.unsqueeze(-1) * h_genes, dim=0)
            pathway_emb[p] = p_emb
            g2p_alpha_dict[p] = (gene_ids, alpha)
        if len(gene_emb_per_pathway) > 0:
            h_stack = torch.stack([h_p for _, h_p in gene_emb_per_pathway])
            # add context here
            p_list = [p for p, _ in gene_emb_per_pathway]
            p_tensor = torch.tensor(p_list, device=h.device)
            num_p = len(p_list)
            # 1. base pooling
            if self.gene_pooling == "mean":
                base_w = torch.ones(num_p, device=h.device)
            elif self.gene_pooling == "size_weighted":
                sizes = torch.tensor(
                    [g.predecessors(p, etype='in_pathway').numel()
                    for p in p_list],
                    device=h.device,
                    dtype=torch.float
                )
                base_w = sizes
            else:
                raise ValueError("Unknown gene_pooling")
            # 2. context
            if context_weights is not None:
                context_w = context_weights[p_tensor]
                w = base_w * (context_w ** self.context_lambda)
            else:
                w = base_w
            # 3. normalize
            w = w / (w.sum() + 1e-8)
            # 4. aggregate
            gene_emb = torch.sum(w.view(-1, 1, 1) * h_stack, dim=0)
        else:
            gene_emb = torch.zeros_like(h)
        return {
            "gene_emb": gene_emb,
            "pathway_emb": pathway_emb,
            "gene_hidden": h,
            "edge_attn": g2p_attn_records,
            "g2p_attn_dict": g2p_alpha_dict
        }

    def forward_mux(self, g, feat, context_weights=None):
        h = self.input_proj(feat)
        h = self.input_norm(h)
        num_genes = g.num_nodes('gene')
        num_pathways = g.num_nodes('pathway')
        edge_mask = g.edges['ppi'].data['pathway_mask']
        take_mask = g.nodes['pathway'].data['take_mask']
        src, dst = g.edges(etype='ppi')
        gene_emb_per_pathway = []
        mux_attn_records = []
        # 1. per-pathway message passing
        for p in range(num_pathways):
            # skip parent pathways
            if not take_mask[p]:
                continue
            eids = torch.nonzero(edge_mask[:, p], as_tuple=True)[0]
            if eids.numel() == 0:
                continue
            subg = dgl.edge_subgraph(g['ppi'], eids, preserve_nodes=True)
            # initialize
            h_p = h.clone()
            # run message passing layers
            for layer in self.layers:
                if self.gnn_type == "gat":
                    h_p = layer(subg, h_p).squeeze(1)
                else:
                    h_p = layer(subg, h_p)
                h_p = F.relu(h_p)
                h_p = self.dropout_layer(h_p)
            gene_emb_per_pathway.append((p, h_p))
        # 2. Mux attention
        gene_emb = torch.zeros(num_genes, h.size(1), device=h.device)
        gene_attn_dict = {}
        for g_id in range(num_genes):
            h_list = []
            p_list = []
            for (p, h_p) in gene_emb_per_pathway:
                h_list.append(h_p[g_id])
                p_list.append(p)
            if len(h_list) == 0:
                gene_emb[g_id] = h[g_id]
                continue
            h_stack = torch.stack(h_list)
            # attention
            scores = self.mux_attn_a(torch.tanh(self.mux_attn_W(h_stack))).squeeze(-1)
            # add context here
            if context_weights is not None:
                pathway_w = context_weights[p_list]
                scores = scores + self.context_lambda * torch.log(pathway_w + 1e-8)
            alpha = torch.softmax(scores, dim=0)
            g_emb = torch.sum(alpha.unsqueeze(-1) * h_stack, dim=0)
            gene_emb[g_id] = g_emb
            record = {
                "pathways": p_list,
                "alpha": alpha
            }
            if pathway_w is not None:
                record["context_w"] = pathway_w
            gene_attn_dict[g_id] = record #(p_list, alpha)
        # 3. gene -> pathway readout
        pathway_emb = torch.zeros(num_pathways, h.size(1), device=h.device)
        g2p_attn_dict = {}
        for (p, h_p) in gene_emb_per_pathway:
            gene_ids = g.predecessors(p, etype='in_pathway')
            if gene_ids.numel() == 0:
                continue
            h_genes = gene_emb[gene_ids]
            scores = self.readout_a(torch.tanh(self.readout_W(h_genes))).squeeze(-1)
            alpha = torch.softmax(scores, dim=0)
            p_emb = torch.sum(alpha.unsqueeze(-1) * h_genes, dim=0)
            pathway_emb[p] = p_emb
            g2p_attn_dict[p] = (gene_ids, alpha)
        # 4. return
        return {
            "gene_emb": gene_emb,
            "pathway_emb": pathway_emb,
            "gene_hidden": h,
            "p2g_attn_dict": gene_attn_dict,
            "g2p_attn_dict": g2p_attn_dict
        }

    def forward_hybrid(self, g, feat, context_weights=None):
        h = self.input_proj(feat)                                                                   
        h = self.input_norm(h)
        num_genes = g.num_nodes('gene')
        num_pathways = g.num_nodes('pathway')
        edge_mask = g.edges['ppi'].data['pathway_mask']
        take_mask = g.nodes['pathway'].data['take_mask']
        src, dst = g.edges(etype='ppi')
        pathway_feat = g.nodes['pathway'].data['feat']
        pathway_query = self.pathway_proj(pathway_feat)
        gene_emb_per_pathway = []
        hybrid_attn_records = []
        # 1 per-pathway processing
        for p in range(num_pathways):
            if not take_mask[p]:
                continue
            # a. Mux GNN backbone
            eids = torch.nonzero(edge_mask[:, p], as_tuple=True)[0]
            if eids.numel() == 0:
                continue
            subg = dgl.edge_subgraph(g['ppi'], eids, preserve_nodes=True)
            h_p = h.clone()
            for layer in self.layers:
                if self.gnn_type == "gat":
                    h_p = layer(subg, h_p).squeeze(1)
                else:
                    h_p = layer(subg, h_p)
                h_new = torch.zeros_like(h_p)
                h_p = F.relu(h_p)
                h_p = self.dropout_layer(h_p)
            # b. Unified refinement
            src_p, dst_p = subg.edges()
            p_q = pathway_query[p]
            p_q_expand = p_q.unsqueeze(0).expand(src_p.shape[0], -1)
            h_src = h_p[src_p]
            h_dst = h_p[dst_p]
            attn_input = torch.cat([h_dst, h_src, p_q_expand], dim=1)
            h_proj = torch.tanh(self.attn_W(attn_input))
            scores = self.attn_a(h_proj).squeeze(-1)
            # mask edges not in pathway
            scores = scores.masked_fill(edge_mask_p == 0, -1e9)
            alpha = dgl.ops.edge_softmax(subg, scores)
            h_refined = torch.zeros_like(h_p)
            h_refined.index_add_(0, dst_p, alpha.unsqueeze(-1) * h_src)
            # residual connection
            h_p = F.relu(h_p + h_refined)
            h_p = self.dropout_layer(h_p)
            gene_emb_per_pathway.append((p, h_p))
            hybrid_attn_records.append((src_p, dst_p, alpha, p))
        # 2. Mux attention
        gene_emb = torch.zeros(num_genes, h.size(1), device=h.device)
        gene_attn_dict = {}
        for g_id in range(num_genes):
            h_list = []
            p_list = []
            for (p, h_p) in gene_emb_per_pathway:
                h_list.append(h_p[g_id])
                p_list.append(p)
            if len(h_list) == 0:
                gene_emb[g_id] = h[g_id]
                continue
            h_stack = torch.stack(h_list)
            scores = self.mux_attn_a(torch.tanh(self.mux_attn_W(h_stack))).squeeze(-1)
            # add context here
            if context_weights is not None:
                pathway_w = context_weights[p_list]
                scores = scores + self.context_lambda * torch.log(pathway_w + 1e-8)
            alpha = torch.softmax(scores, dim=0)
            g_emb = torch.sum(alpha.unsqueeze(-1) * h_stack, dim=0)
            gene_emb[g_id] = g_emb
            gene_attn_dict[g_id] = (p_list, alpha)
        # 3. gene -> pathway
        pathway_emb = torch.zeros(num_pathways, h.size(1), device=h.device)
        g2p_attn_dict = {}
        for (p, _) in gene_emb_per_pathway:
            gene_ids = g.predecessors(p, etype='in_pathway')
            if gene_ids.numel() == 0:
                continue
            h_genes = gene_emb[gene_ids]
            scores = self.readout_a(torch.tanh(self.readout_W(h_genes))).squeeze(-1)
            alpha = torch.softmax(scores, dim=0)
            p_emb = torch.sum(alpha.unsqueeze(-1) * h_genes, dim=0)
            pathway_emb[p] = p_emb
            g2p_attn_dict[p] = (gene_ids, alpha)
        # 4. Return
        return {
            "gene_emb": gene_emb,
            "pathway_emb": pathway_emb,
            "gene_hidden": h,
            "edge_attn": hybrid_attn_records,
            "p2g_attn_dict": gene_attn_dict,
            "g2p_attn_dict": g2p_attn_dict
        }

    def forward(self, g, feat, context_weights):
        if self.mode == "baseline":
            return self.forward_baseline(g, feat, context_weights)
        elif self.mode == "unified":
            return self.forward_unified(g, feat, context_weights)
        elif self.mode == "mux":
            return self.forward_mux(g, feat, context_weights)
        elif self.mode == "hybrid":
            return self.forward_hybrid(g, feat, context_weights)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class PathwayHierarchyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        # attention for child -> parent
        self.attn_W_up = nn.Linear(2 * dim, dim)
        self.attn_a_up = nn.Linear(dim, 1)
        # attention for parent -> child
        self.attn_W_down = nn.Linear(2 * dim, dim)
        self.attn_a_down = nn.Linear(dim, 1)

    def forward(self, g, pathway_emb):
        # Debug: check hierarchy edges
        #print(f"Child of edges: ", g.num_edges('child_of'))
        g.nodes['pathway'].data['h'] = pathway_emb
        # 1. child -> parent
        src_c, dst_p = g.edges(etype='child_of')
        h_src = pathway_emb[src_c] 
        h_dst = pathway_emb[dst_p]
        attn_input = torch.cat([h_dst, h_src], dim=1)
        h_proj = torch.tanh(self.attn_W_up(attn_input))
        scores_up = self.attn_a_up(h_proj).squeeze(-1)
        alpha_up = dgl.ops.edge_softmax(g['child_of'], scores_up)
        h_up = torch.zeros_like(pathway_emb)
        h_up.index_add_(0, dst_p, alpha_up.unsqueeze(-1) * h_src)
        # 2. parent -> child
        src_p, dst_c = g.edges(etype='parent_of')
        h_src = pathway_emb[src_p]
        h_dst = pathway_emb[dst_c]
        attn_input = torch.cat([h_dst, h_src], dim=1)
        h_proj = torch.tanh(self.attn_W_down(attn_input))
        scores_down = self.attn_a_down(h_proj).squeeze(-1)
        alpha_down = dgl.ops.edge_softmax(g['parent_of'], scores_down)
        h_down = torch.zeros_like(pathway_emb)
        h_down.index_add_(0, dst_c, alpha_down.unsqueeze(-1) * h_src)
        # 3. combine + transform
        h_new = h_up + h_down
        h_new = self.W(h_new)
        # residual connection
        return pathway_emb + h_new, {
            "child_to_parent": (src_c, dst_p, alpha_up),
            "parent_to_child": (src_p, dst_c, alpha_down)
        }

class MHHGNN(nn.Module):
    """
    Multiplex Heterogeneous Hierarchical GNN for gene-pathway networks
    """
    def __init__(
        self,
        num_organs: int,
        num_cells: int,
        num_diseases: int,
        num_stimuli: int,
        gnn_type: str = "gcn",
        num_layers: int = 2,
        in_dim: int = 128,
        hidden_dim: int = 128,
        emb_dim: int = 64,
        attn_dim: int = 16,
        dropout_prob: float = 0.2,
        device='cpu'
    ):
        super().__init__()
        # Save hyperparameters
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.organ_lookup = nn.Embedding(num_organs, emb_dim)
        self.cell_lookup = nn.Embedding(num_cells, emb_dim)
        self.disease_lookup = nn.Embedding(num_diseases, emb_dim)
        self.stimulus_lookup = nn.Embedding(num_stimuli, emb_dim)
        self.dropout_prob = dropout_prob
        self.device = device
        self.use_context_in_encoder = True
        self.use_soft_feedback = True
        # Gene encoder with PPI message passing + attention
        self.encoder = GeneEncoder(
            gnn_type=self.gnn_type,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=emb_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        # Hierarchy layer with pathway hierarchy
        self.hierarchy_layer = PathwayHierarchyLayer(
        )
        # Context MLP
        self.context_mlp = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )
        # loss
        self.loss_fn = PathwayNegativeSamplingLoss(device=device)
        # Dropout for final embeddings if needed
        self.final_dropout = nn.Dropout(dropout_prob)

    def get_context_vector(self, ctx, device):
        organ_vec = self.organ_lookup(
            torch.tensor(ctx.organ_id, device=device)
        )
        cell_vec = self.cell_lookup(
            torch.tensor(ctx.cell_type_id, device=device)
        )
        disease_vec = self.disease_lookup(
            torch.tensor(ctx.disease_id, device=device)
        )
        stimulus_vec = self.stimulus_lookup(
            torch.tensor(ctx.stimulus_id, device=device)
        )
        context_vec = organ_vec + cell_vec + disease_vec + stimulus_vec
        context_vec = context_vec / context_vec.norm()
        return context_vec

    def apply_context_lookup(self, pathway_emb, ctx):
        """
        pathway_emb: (num_pathways, emb_dim)
        """
        #context_vec = self.context_lookup(torch.tensor(context_id, device=pathway_emb.device)) # emb_dim
        context_vec = self.get_context_vector(ctx, pathway_emb.device)
        print(f"Context geo id: ", ctx.geo_id)
        print(f"Context vector norm:", context_vec.norm().item())
        context_vec = context_vec.unsqueeze(0) # (1, emb_dim)
        weights = torch.sum(pathway_emb * context_vec, dim=1, keepdim=True)
        weights = torch.sigmoid(weights)
        print(
            f"Context weights stats:",
            weights.min().item(),
            weights.max().item(),
            weights.mean().item()
        )
        pathway_emb = pathway_emb * weights
        return pathway_emb, weights

    def apply_context_mlp(self, pathway_emb, ctx):
        #context_vec = self.context_lookup(torch.tensor(context_id, device=pathway_emb.device))
        context_vec = self.get_context_vector(ctx, pathway_emb.device)
        context_vec = context_vec.unsqueeze(0).repeat(
            pathway_emb.size(0), 1
        )
        combined = torch.cat([pathway_emb, context_vec], dim=1)
        weights = self.context_mlp(combined)
        weights = torch.sigmoid(weights)
        pathway_emb = pathway_emb * weights
        return pathway_emb, weights

    def _build_unified_attention(self, enc_out, g):
        attn = {
            "gene_to_pathway": None,
            "pathway_to_gene": None,
            "gene_to_gene": None
        }
        # gene -> pathway
        if "g2p_edge_index" in enc_out:
            # baseline
            src, dst = enc_out["g2p_edge_index"]
            attn["gene_to_pathway"] = {
                "index": (src, dst),
                "alpha": enc_out["g2p_edge_attn"]
            }
        elif "g2p_attn_dict" in enc_out:
            # unified / mux / hybrid
            g_list, p_list, a_list = [], [], []
            for p, (gene_ids, alpha) in enc_out["g2p_attn_dict"].items():
                g_list.append(gene_ids)
                p_list.append(torch.full_like(gene_ids, p))
                a_list.append(alpha)
            attn["gene_to_pathway"] = {
                "index": (torch.cat(g_list), torch.cat(p_list)),
                "alpha": torch.cat(a_list)
            }
        # pathway -> gene
        if "p2g_edge_index" in enc_out:
            # baseline
            src, dst = enc_out["p2g_edge_index"]
            attn["pathway_to_gene"] = {
                "index": (src, dst),
                "alpha": enc_out["p2g_edge_attn"]
            }
        elif "p2g_attn_dict" in enc_out:
            # mux / hybrid
            p_list_all, g_list_all, a_list_all = [], [], []
            for g_id, record in enc_out["p2g_attn_dict"].items():
                # support tuple and dict
                if isinstance(record, dict):
                    p_list = record["pathways"]
                    alpha = record["alpha"]
                else:
                    p_list, alpha = record
                p_tensor = torch.tensor(p_list, device=alpha.device)
                g_tensor = torch.full_like(p_tensor, g_id)
                p_list_all.append(p_tensor)
                g_list_all.append(g_tensor)
                a_list_all.append(alpha)
            attn["pathway_to_gene"] = {
                "index": (torch.cat(p_list_all), torch.cat(g_list_all)),
                "alpha": torch.cat(a_list_all)
            }
        # gene -> gene
        if "edge_attn" in enc_out:
            src_list, dst_list, p_list, a_list = [], [], [], []
            for (src, dst, alpha, p) in enc_out["edge_attn"]:
                src_list.append(src)
                dst_list.append(dst)
                p_list.append(torch.full_like(src, p))
                a_list.append(alpha)
            attn["gene_to_gene"] = {
                "index": (
                    torch.cat(src_list),
                    torch.cat(dst_list),
                    torch.cat(p_list)
                ),
                "alpha": torch.cat(a_list)
            }
        return attn

    def forward(self, g, ctx, sampler=None, compute_loss=False):
        """
        Forward pass:
        - g: PPI reactome pathway graph (DGLGraph)
        - feat: gene features (num_genes, in_dim)
        - pathway_nodes: dict {pathway_id: [gene_ids]}
        - gene2pathways: dict {gene_id: [pathway_ids]}
        """
        gene_feat = g.nodes['gene'].data['feat']
        num_genes = g.num_nodes('gene')
        num_pathways = g.num_nodes('pathway')
        # Encode genes and pathways
        enc_out = self.encoder(g, gene_feat)
        attn = self._build_unified_attention(enc_out, g)
        gene_emb = enc_out["gene_emb"]
        pathway_emb = enc_out["pathway_emb"]
        gene_hidden = enc_out["gene_hidden"]
        print("Pathway emb BEFORE hierarchy:", pathway_emb.shape)
        # Hierarchy propagation
        pathway_emb, hierarchy_attn = self.hierarchy_layer(g, pathway_emb)
        print("Hierarchy propagated emb stats: min {:.4f}, max {:.4f}, mean {:.4f}".format(
            pathway_emb.min().item(),
            pathway_emb.max().item(),
            pathway_emb.mean().item()
        ))
        print("Pathway emb AFTER hierarchy:", pathway_emb.shape)
        # Apply context weighting
        pathway_emb, context_weights = self.apply_context_lookup(
            pathway_emb,
            ctx
        )
        # Pack embeddings for loss
        embeddings = {
            "gene": gene_emb,
            "pathway": pathway_emb
        }
        #id2pathway = g.graph_data['id2pathway']
        #topk = torch.topk(context_weights.squeeze(), 10)
        #print(f"The top 10 pathways ranked by weight")
        #for idx, w in zip(topk.indices, topk.values):
        #    pid = idx.item()
        #    pname = id2pathway.get(pid, "UNKNOWN")
        #    print(f"{pid:4d} | {pname:20s} | weight={w.item():.4f}")
        #rand_idx = torch.randint(0, context_weights.shape[0], (10,))
        #print(f"The 10 random pathways")
        #for idx in rand_idx:
        #    pid = idx.item()
        #    pname = id2pathway[pid]
        #    w = context_weights[pid].item()
        #    print(f"{pid:4d} | {pname:20s} | weight={w:.4f}")
        # If no loss -> inference mode
        if not compute_loss:
            return {
                "gene_emb": gene_emb,
                "pathway_emb": pathway_emb,
                "gene_hidden": gene_hidden,
                "p2g_src": p2g_src,
                "p2g_dst": p2g_dst,
                "p2g_attn": p2g_attn,
                "g2p_src": g2p_src,
                "g2p_dst": g2p_dst,
                "g2p_attn": g2p_attn,
                "context_weights": context_weights
            }

        # Sampling
        sample_dict = sampler.sample_pos_neg(ctx)
        # Loss
        loss = self.loss_fn(embeddings, sample_dict)
        # For now, just return gene embeddings and attention maps
        return loss, {
            "gene_emb": gene_emb,
            "pathway_emb": pathway_emb,
            "context_weights": context_weights
        }

