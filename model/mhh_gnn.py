import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn import GraphConv, GATConv, GINConv

import math
from loss import PathwayNegativeSamplingLoss

class GeneEncoder(nn.Module):
    """
    Gene-level GNN encoder using local PPI subgraphs. Supporting GCN, GAT, and GIN
    """
    def __init__(
        self,
        context_info,
        gnn_type: str,
        in_dim: int, # create before model creation: g.nodes['gene'].data['feat'].shape[1]
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
        self.in_dim = in_dim
        self.attn_dim = attn_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.pathway_feat_dim = pathway_feat_dim
        self.mode = mode # baseline / unified / mux / hybrid
        
        # Context integration
        # 1. Context ID embeddings
        self.counts = context_info['counts']
        self.organ_emb = nn.Embedding(self.counts['organ'], self.hidden_dim)
        self.cell_emb = nn.Embedding(self.counts['cell'], self.hidden_dim)
        self.disease_emb = nn.Embedding(self.counts['disease'], self.hidden_dim)
        self.stimulus_emb = nn.Embedding(self.counts['stimulus'], self.hidden_dim)
        # 2. Projection to merge all 4 context IDs into one vector
        self.context_merge = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        # 3. Context gating
        self.context_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Dimension projection:
        # 1. Gene projection
        self.gene_input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.gene_input_norm = nn.LayerNorm(self.hidden_dim)
        self.gene_output_proj = nn.Linear(self.hidden_dim, self.out_dim)
        # 2. Pathway projection
        self.pathway_input_proj = nn.Linear(self.pathway_feat_dim, self.hidden_dim)
        self.pathway_input_norm = nn.LayerNorm(self.hidden_dim)
        self.pathway_output_proj = nn.Linear(self.hidden_dim, self.out_dim)
        
        # Layers
        # 1. GNN layers for message passing
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self.build_layer(self.hidden_dim, self.hidden_dim))
        # Norm of GNN output
        self.gnn_output_norm = nn.LayerNorm(self.hidden_dim)
        # 2. Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout_prob)
        # 3. The Hierarchy refiner layer (pathway -> pathway)
        self.hierarchy_layer = PathwayHierarchyLayer(dim=self.hidden_dim, attn_dim=self.attn_dim)
        # 4. The pathway attention layer for depth influence
        #self.pathway_attn = PathwayAttentionLayer(hidden_dim=hidden_dim)
        #self.pathway_attn = MultiPathwayAttentionLayer(hidden_dim=hidden_dim)
        self.pathway_attn = UnifiedPathwayAttentionLayer(hidden_dim=self.hidden_dim, attn_dim=self.attn_dim)
        
        # Attention for baseline
        # Gene-pathway attention
        self.gene_attn_W = nn.Linear(self.hidden_dim * 3, self.attn_dim, bias=False)
        self.gene_attn_a = nn.Linear(self.attn_dim, 1, bias=False)
        # Pathway-gene attention
        self.pathway_attn_W = nn.Linear(self.hidden_dim * 3, self.attn_dim, bias=False)
        self.pathway_attn_a = nn.Linear(self.attn_dim, 1, bias=False)
        # Instance attention for Gene after message passing
        self.instance_attn = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Attention for unified
        # Relation attention
        self.rel_attn_W = nn.Linear(self.hidden_dim * 4, self.attn_dim, bias=False)
        self.rel_attn_a = nn.Linear(self.attn_dim, 1, bias=False)
        # Readout attention
        self.readout_W = nn.Linear(self.hidden_dim * 2, self.attn_dim, bias=False)
        self.readout_a = nn.Linear(self.attn_dim, 1, bias=False)
        # Pathway attetnion
        self.pathway_attn_W = nn.Linear(self.hidden_dim * 2, self.attn_dim, bias=False)
        self.pathway_attn_a = nn.Linear(self.attn_dim, 1, bias=False)
        
        # Mux attention
        self.mux_attn_W = nn.Linear(self.hidden_dim * 2, self.attn_dim, bias=False)
        self.mux_attn_a = nn.Linear(self.attn_dim, 1, bias=False)

        # Normalization of embedding
        # Baseline
        self.g2p_norm = nn.LayerNorm(self.hidden_dim)
        self.instance_output_norm = nn.LayerNorm(self.hidden_dim)
        self.p_q_norm = nn.LayerNorm(self.hidden_dim)
        self.attn_input_norm = nn.LayerNorm(self.hidden_dim * 3)
        self.norm_p2g = nn.LayerNorm(self.hidden_dim)
        # Unified
        self.gene_attn_norm = nn.LayerNorm(self.hidden_dim)
        self.attn_input_norm_unified = nn.LayerNorm(self.hidden_dim * 4)
        self.h_readout_norm = nn.LayerNorm(self.hidden_dim)
        self.h_readout_input_norm = nn.LayerNorm(self.hidden_dim)
        self.ppi_norm = nn.LayerNorm(self.hidden_dim)
        self.seed_norm = nn.LayerNorm(self.hidden_dim)
        self.pathway_fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.attn_in_norm = nn.LayerNorm(self.hidden_dim * 2)
        self.p2g_norm = nn.LayerNorm(self.hidden_dim)
        # Mux
        self.mux_input_norm = nn.LayerNorm(self.hidden_dim * 2)
        self.refined_gene_norm = nn.LayerNorm(self.hidden_dim)
        self.readout_input_norm = nn.LayerNorm(self.hidden_dim * 2)
        self.readout_output_norm = nn.LayerNorm(self.hidden_dim)

        # Hybrid
        self.edge_attn_norm = nn.LayerNorm(self.hidden_dim * 3)

        # Isolated gene aligner
        self.isolated_aligner = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Auto Fusion
        # Baseline / Unified
        self.baseline_fuser = AutoFusionModule(self.hidden_dim, num_streams=2)
        # Mux / Hybrid
        self.mux_fuser = AutoFusionModule(self.hidden_dim, num_streams=3)

    def get_context_vector(self, ctx):
        device = self.context_merge.weight.device
        # Get 4 context IDs
        organ_id = torch.tensor([ctx.organ_id], device=device)
        cell_id = torch.tensor([ctx.cell_type_id], device=device)
        disease_id = torch.tensor([ctx.disease_id], device=device)
        stimulus_id = torch.tensor([ctx.stimulus_id], device=device)

        # Lookup the individual embeddings
        o_emb = self.organ_emb(organ_id)
        c_emb = self.cell_emb(cell_id)
        d_emb = self.disease_emb(disease_id)
        s_emb = self.stimulus_emb(stimulus_id)

        # Concatenate the embeddings
        combined = torch.cat([o_emb, c_emb, d_emb, s_emb], dim=-1)

        # Project to a single vector
        context_vec = torch.relu(self.context_merge(combined))
        context_vec = self.context_norm(context_vec)

        # Apply non-linearity
        return context_vec # 1D vector: hidden_dim


    def build_layer(self, in_dim, out_dim):
        if self.gnn_type == "gcn":
            return GraphConv(in_dim, out_dim, norm="both", allow_zero_in_degree=True)
        elif self.gnn_type == "gat":
            return GATConv(in_dim, out_dim, num_heads=1, concat=False, feat_drop=0.0, attn_drop=0.0, allow_zero_in_degree=True)
        elif self.gnn_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINConv(mlp, aggregator_type="sum")
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

    def forward_baseline(self, g, feat, ctx):
        """
        Input:
            g: DGLGraph (can be full PPI or k-hop subgraph)
            feat: gene_dim, in_dim
        """
        device = feat.device

        # Merge the context (4 IDs) into one unified context representation
        ctx_vec = self.get_context_vector(ctx) # 1, hidden_dim
        # Generate the bio-filter (Gate)
        gate = self.context_gate(ctx_vec) # 1, hidden_dim

        # 1. Initial projection & gating
        h = self.gene_input_proj(feat) # gene_dim, hidden_dim
        h = self.gene_input_norm(h)
        h = h * gate # gene_dim, hidden_dim

        # Pre-computed subgraphs
        batched_ppi = g.graph_data['batched_ppi_graph'].to(device)

        # Pick the active gene instances
        global_nids = batched_ppi.ndata[dgl.NID]
        h_batched = h[global_nids] # gene instances, hidden_dim

        # Identify which genes are processed in the pathways
        has_pathway_mask = torch.zeros(h.size(0), 1, device=device)
        has_pathway_mask.index_fill_(0, global_nids, 1.0)

        # 2. Run GNN computation layers for all leaf pathways
        for layer in self.layers:
            h_batched = layer(batched_ppi, h_batched)
            if self.gnn_type == "gat": 
                #h_batched = h_batched.squeeze(1)
                h_batched = h_batched.view(h_batched.shape[0], -1)
            h_batched = F.relu(self.dropout_layer(h_batched)) # gene instances, hidden_dim

        # Normalization after GNN layer
        h_batched_temp = self.gnn_output_norm(h_batched)

        # Importance scores for every gene instances
        # Since each gene would show up multiple times in different pathways to gather message from genes in the same pathway, attention-based pooling using context is applied here to weight context-related pathway genes
        instance_scores = self.instance_attn(torch.cat([
            h_batched_temp,
            ctx_vec.expand(h_batched.size(0), -1)
        ], dim=-1))
        # Find max score per gene
        gene_max = torch.full((h.size(0), 1), -1e9, device=device)
        gene_max.index_reduce_(0, global_nids, instance_scores, reduce='amax', include_self=False)
        # Segment Softmax handling multi-instance genes
        shifted_scores = instance_scores - gene_max[global_nids]
        exp_scores = torch.exp(shifted_scores)
        # Segment sum
        sum_exp = torch.zeros(h.size(0), 1, device=device).index_add_(0, global_nids, exp_scores)
        instance_weights = exp_scores / (sum_exp[global_nids] + 1e-8)
        # Weighted sum for h_updated
        h_updated = torch.zeros_like(h)
        h_updated.index_add_(0, global_nids, instance_weights * h_batched) # gene_dim, hidden_dim
        h_updated = self.instance_output_norm(h_updated)

        # 3. Context-aware readout Gene -> Pathway
        pathway_feat = g.nodes['pathway'].data['feat'] # pathway_dim, pathway_feat_dim
        pathway_query = self.pathway_input_proj(pathway_feat) # pathway_dim, hidden_dim
        pathway_query = self.pathway_input_norm(pathway_query) # pathway_dim, hidden_dim
        pathway_query = pathway_query * gate # pathway_dim, hidden_dim

        # Get pathway ID for every node in h_batched
        pathway_indices = g.graph_data['pathway_indices'].to(device)
        pathway_ids_for_nodes = torch.repeat_interleave(
            pathway_indices,
            batched_ppi.batch_num_nodes()
        )

        # Attention components
        p_q_normed = self.p_q_norm(pathway_query)
        p_q_expanded = p_q_normed[pathway_ids_for_nodes] # gene instances, hidden_dim
        c_v_expanded = ctx_vec.expand(h_batched.size(0), -1) # gene instances, hidden_dim

        attn_input = torch.cat([h_batched_temp, p_q_expanded, c_v_expanded], dim=1) # gene instances, 3 * hidden_dim
        attn_input = self.attn_input_norm(attn_input)
        scores = self.gene_attn_a(torch.tanh(self.gene_attn_W(attn_input))).squeeze(-1) # gene instances

        # Normalize scores per pathway
        alpha_g2p = dgl.ops.softmax_nodes(batched_ppi, scores) # gene instances

        # Weighted sum into pathway embeddings: for each pathway i, h_p,i = sum_j (alpha_ji * h_g,j)
        pathway_emb = pathway_query.clone()
        pathway_emb.index_add_(0, pathway_ids_for_nodes, h_batched * alpha_g2p.unsqueeze(-1)) # pathway_dim, hidden_dim
        pathway_emb = self.g2p_norm(pathway_emb)

        # 4. Hierarchy refinement
        pathway_emb, attn_data = self.hierarchy_layer(g, pathway_emb, ctx_vec)

        # 5. Pathway -> Gene
        h_pathway_context, alpha_p2g = self.pathway_attn(
            g, pathway_emb=pathway_emb, gene_emb=h_updated, ctx_vec=ctx_vec
        )
        
        # 6. Adaptive Auto-Fusion with Symmetrical Imputation and Vector Alignment
        # Vector alignment of isolated genes in parent pathways
        h_raw_aligned = self.isolated_aligner(h)
        # Ensure both inputs are full without zero-vectors
        h_ppi_stream = has_pathway_mask * h_updated + (1. - has_pathway_mask) * h_raw_aligned
        h_path_stream = has_pathway_mask * h_pathway_context + (1. - has_pathway_mask) * h_raw_aligned # gene_dim, hidden_dim
        gene_emb, auto_loss = self.baseline_fuser(h_ppi_stream, h_path_stream) # gene_dim, hidden_dim
        gene_emb = self.norm_p2g(gene_emb)

        # Final projection to out_dim
        final_gene_emb = self.gene_output_proj(gene_emb) # gene_dim, out_dim
        final_pathway_emb = self.pathway_output_proj(pathway_emb) # pathway_dim, out_dim

        # Prepare attention dictionary
        global_gene_ids = batched_ppi.ndata[dgl.NID]
        global_pathway_ids = pathway_indices.repeat_interleave(batched_ppi.batch_num_nodes())
        g2p_attn_dict = {
            'src': global_gene_ids.cpu(), 'dst': global_pathway_ids.cpu(), 'weights': alpha_g2p.detach().cpu()
        }

        src_p, dst_g = g.edges(etype=('pathway', 'has_gene', 'gene'))
        p2g_attn_dict = {
            'src': src_p.cpu(), 'dst': dst_g.cpu(), 'weights': alpha_p2g.detach().cpu()
        }

        return {
            "gene_emb": final_gene_emb,
            "pathway_emb": final_pathway_emb,
            "context_vec": ctx_vec,
            "auto_loss": auto_loss,
            "p2g_attn_dict": p2g_attn_dict,
            "g2p_attn_dict": g2p_attn_dict
        }


    def forward_unified(self, g, feat, ctx):
        # Unified Multiplex + Attention
        device = feat.device

        # Context generation
        ctx_vec = self.get_context_vector(ctx) # 1, hidden_dim
        gate = self.context_gate(ctx_vec) # 1, hidden_dim

        # 1. Initial projection & Gating
        h = self.gene_input_proj(feat) # gene_dim, hidden_dim
        h = self.gene_input_norm(h)
        h = h * gate # gene_dim, hidden_dim

        pathway_feat = g.nodes['pathway'].data['feat'] # pathway_dim, pathway_feat_dim
        # Project pathway features and apply gate
        pathway_query = self.pathway_input_proj(pathway_feat)
        pathway_query = self.pathway_input_norm(pathway_query)
        pathway_query = pathway_query * gate # pathway_dim, hidden_dim
        
        num_pathways = g.num_nodes('pathway')
        # Get the batched PPI subgraph
        batched_ppi = g.graph_data['batched_ppi_graph'].to(device)
        # IDs of leaf pathways
        pathway_indices = g.graph_data['pathway_indices'].to(device)

        # Pick the active gene instances
        global_nids = batched_ppi.ndata[dgl.NID]
        has_pathway_mask = torch.zeros(h.size(0), 1, device=device)
        has_pathway_mask.index_fill_(0, global_nids, 1.0)

        # Repeat each pathway query by the number of edges in its corresponding PPI subgraph
        pathway_query_normed = self.p_q_norm(pathway_query)
        active_pathway_query = pathway_query_normed[pathway_indices]
        p_q_per_edge = torch.repeat_interleave(active_pathway_query,
                batched_ppi.batch_num_edges(), dim=0)

        # 2. Unified PPI attention, assign unique importance score for evey PPI tailored to the pathway and context 
        src_p, dst_p = batched_ppi.edges()
        # Original global ID for the loca nodes in subgraph
        original_src = batched_ppi.ndata[dgl.NID][src_p]
        original_dst = batched_ppi.ndata[dgl.NID][dst_p]

        h_normed = self.gene_attn_norm(h)
        # dst as Query to receive message and src as Key to provide content
        attn_input = torch.cat([
            h_normed[original_dst],
            h_normed[original_src],
            p_q_per_edge,
            ctx_vec.expand(p_q_per_edge.size(0), -1)
        ], dim=1)

        attn_input = self.attn_input_norm_unified(attn_input)
        scores = self.rel_attn_a(torch.tanh(self.rel_attn_W(attn_input))).squeeze(-1)
        alpha = dgl.ops.edge_softmax(batched_ppi, scores)

        # Aggregate PPI info
        batched_ppi.edata['m'] = alpha.unsqueeze(-1) * h[original_src]
        batched_ppi.update_all(fn.copy_e('m', 'msg'), fn.sum('msg', 'h_p'))

        # 3. Readout: vector per active pathway subgraph
        h_p_refined = batched_ppi.ndata['h_p'] # gene instances, hidden_dim
        h_p_refined = self.h_readout_norm(h_p_refined)

        # Container for PPI stream
        ppi_counts = torch.zeros(h.size(0), 1, device=device)
        ppi_counts.index_add_(0, global_nids, torch.ones(h_p_refined.size(0), 1, device=device))

        h_ppi_global = torch.zeros(h.size(0), self.hidden_dim, device=device)
        h_ppi_global.index_add_(0, global_nids, h_p_refined)
        h_ppi_global = h_ppi_global / (ppi_counts + 1e-8)
        h_ppi_global = self.ppi_norm(h_ppi_global)

        ctx_expand_nodes = ctx_vec.expand(h_p_refined.size(0), -1) # gene instances, hidden_dim

        # Attention: which genes are the stars of this pathway in this context?
        readout_input = torch.cat([h_p_refined, ctx_expand_nodes], dim=1) # gene instances, hidden_dim * 2
        readout_input = self.readout_input_norm(readout_input)
        readout_scores = self.readout_a(torch.tanh(self.readout_W(readout_input))).squeeze(-1) # gene instances

        # Softmax only within the boundaries of each specific pathway in the batch
        alpha_readout = dgl.softmax_nodes(batched_ppi, readout_scores) # gene_in_pathway_num

        # Weighted sum of genes, one vector per pathway
        batched_ppi.ndata['weighted_genes'] = alpha_readout.unsqueeze(-1) * h_p_refined
        active_seeds = dgl.readout_nodes(batched_ppi, 'weighted_genes', op='sum') # leaf_pathway_dim, hidden_dim

        # Preparation for hierarchy pathway aggregation
        pathway_gene_seeds = torch.zeros(num_pathways, self.hidden_dim, device=device)
        pathway_gene_seeds[pathway_indices] = active_seeds
        seeds_normed = self.seed_norm(pathway_gene_seeds)
        seeds_normed = seeds_normed * (pathway_gene_seeds.abs().sum(dim=-1, keepdim=True) > 0).float()

        # Add metadata + gene evidence
        pathway_emb = pathway_query.clone()
        pathway_emb = pathway_emb + seeds_normed
        pathway_emb = self.pathway_fusion_norm(pathway_emb)

        # Hierarchy refinement
        pathway_emb, attn_data = self.hierarchy_layer(g, pathway_emb, ctx_vec)

        # 4. Hierarchical Gene Refinement (Pathway -> Gene)
        gene_from_pathway, alpha_p2g = self.pathway_attn(
            g, pathway_emb=pathway_emb, gene_emb=h_normed, ctx_vec=ctx_vec
        )

        # e. Adaptive Auto-Fusion with Symmetrical Imputation and Vector Alignment
        h_raw_aligned = self.isolated_aligner(h)
        # Ensure both inputs are full without zero-vectors
        h_ppi_stream = has_pathway_mask * h_ppi_global + (1. - has_pathway_mask) * h_raw_aligned
        h_path_stream = has_pathway_mask * gene_from_pathway + (1. - has_pathway_mask) * h_raw_aligned # gene_dim, hidden_dim
        gene_emb, auto_loss = self.baseline_fuser(h_ppi_stream, h_path_stream) # gene_dim, hidden_dim
        gene_emb = self.norm_p2g(gene_emb)

        # Final projection to out_dim
        final_gene_emb = self.gene_output_proj(gene_emb) # gene_dim, out_dim
        final_pathway_emb = self.pathway_output_proj(pathway_emb) # pathway_dim, out_dim

        # Prepare attention dictionary
        src_g, dst_p = g.edges(etype='in_pathway')
        p2g_attn_dict = {
            'src': src_g.cpu(), 'dst': dst_p.cpu(), 'weights': alpha_p2g.detach().cpu()
        }
        global_gene_ids = batched_ppi.ndata[dgl.NID]
        global_pathway_ids = pathway_indices.repeat_interleave(batched_ppi.batch_num_nodes())
        g2p_attn_dict = {
            'src': global_gene_ids.cpu(), 'dst': global_pathway_ids.cpu(), 'weights': alpha_readout.detach().cpu()
        }

        return {
            "gene_emb": final_gene_emb,
            "pathway_emb": final_pathway_emb,
            "context_vec": ctx_vec,
            "auto_loss": auto_loss,
            "g2p_attn_dict": g2p_attn_dict,
            "p2g_attn_dict": p2g_attn_dict
        }


    def forward_mux(self, g, feat, ctx):
        device = feat.device
        
        # Context generation
        ctx_vec = self.get_context_vector(ctx) # 1, hidden_dim
        gate = self.context_gate(ctx_vec) # 1, hidden_dim

        # 1. Initial projection & Gating
        h = self.gene_input_proj(feat) # gene_num, hidden_dim
        h = self.gene_input_norm(h)
        h = h * gate

        # Get the batched PPI subgraph
        batched_ppi = g.graph_data['batched_ppi_graph'].to(device)
        # IDs of leaf pathways
        pathway_indices = g.graph_data['pathway_indices'].to(device)
        
        # Map global gene features to the nodes in the batched subgraph 
        global_nids = batched_ppi.ndata[dgl.NID]
        h_batched = h[global_nids] # genes in batch, hidden_dim

        # Identify which genes are processed in the pathways
        has_pathway_mask = torch.zeros(h.size(0), 1, device=device)
        has_pathway_mask.index_fill_(0, global_nids, 1.0)

        # 2. Run GNN layers to do message passing
        for layer in self.layers:
            h_batched = layer(batched_ppi, h_batched)
            if self.gnn_type == "gat":
                #h_batched = layer(batched_ppi, h_batched).squeeze(1)
                h_batched = h_batched.view(h_batched.shape[0], -1)

            h_batched = F.relu(self.dropout_layer(h_batched))

        # Normalization after GNN layer
        h_batched_temp = self.gnn_output_norm(h_batched) # gene instances, hidden_dim

        # Importance scores for every gene instances
        # Since each gene would show up multiple times in different pathways to gather message from genes in the same pathway, attention-based pooling using context is applied here to weight context-related pathway genes
        instance_scores = self.instance_attn(torch.cat([
            h_batched_temp,
            ctx_vec.expand(h_batched.size(0), -1)
        ], dim=-1))
        # Find max score per gene
        gene_max = torch.full((h.size(0), 1), -1e9, device=device)
        gene_max.index_reduce_(0, global_nids, instance_scores, reduce='amax', include_self=False)
        # Segment Softmax handling multi-instance genes
        shifted_scores = instance_scores - gene_max[global_nids]
        exp_scores = torch.exp(shifted_scores)
        # Segment sum
        sum_exp = torch.zeros(h.size(0), 1, device=device).index_add_(0, global_nids, exp_scores)
        instance_weights = exp_scores / (sum_exp[global_nids] + 1e-8)
        # Weighted sum for h_updated
        h_updated = torch.zeros_like(h)
        h_updated.index_add_(0, global_nids, instance_weights * h_batched) # gene_dim, hidden_dim
        h_updated = self.instance_output_norm(h_updated)

        # 3. Contextual Mux Attention
        # Expand context to match gene in the batch
        ctx_expand = ctx_vec.expand(h_batched.size(0), -1)

        # Compute Mux attention scores
        mux_input = torch.cat([h_batched_temp, ctx_expand], dim=1)
        mux_input = self.mux_input_norm(mux_input)
        mux_scores = self.mux_attn_a(torch.tanh(self.mux_attn_W(mux_input))).squeeze(-1)

        # Softmax over each gene
        alpha_mux = dgl.softmax_nodes(batched_ppi, mux_scores)

        # Refined tensor
        refined_gene_emb = torch.zeros_like(h)
        weighted_h = h_batched * alpha_mux.unsqueeze(-1)
        refined_gene_emb.index_add_(0, global_nids, weighted_h)
        refined_gene_emb = self.refined_gene_norm(refined_gene_emb)

        # 4. Gene -> Pathway readout
        h_pull = h_batched_temp.clone()

        # Compute readout attention scores
        readout_input = torch.cat([h_pull, ctx_expand], dim=1)
        readout_input = self.readout_input_norm(readout_input)
        readout_scores = self.readout_a(torch.tanh(self.readout_W(readout_input))).squeeze(-1)

        # Softmax per pathway
        alpha_readout = dgl.softmax_nodes(batched_ppi, readout_scores)

        # Aggregate to get pathway embeddings
        batched_ppi.ndata['weighted_final'] = h_batched * alpha_readout.unsqueeze(-1)
        pathway_emb_batch = dgl.sum_nodes(batched_ppi, 'weighted_final')
        pathway_emb_batch = self.readout_output_norm(pathway_emb_batch)

        # Scatter back to global pathway tensor
        pathway_feat = g.nodes['pathway'].data['feat']
        pathway_emb = self.pathway_input_proj(pathway_feat)
        pathway_emb = self.pathway_input_norm(pathway_emb)
        pathway_emb = pathway_emb * gate

        # Add gene contribution to the leaf pathways
        pathway_emb.index_add_(0, pathway_indices, pathway_emb_batch)
        pathway_emb = self.g2p_norm(pathway_emb)

        # Hierarchy refinement
        pathway_emb, attn_data = self.hierarchy_layer(g, pathway_emb, ctx_vec)

        h_pathway_context, alpha_p2g = self.pathway_attn(g, pathway_emb=pathway_emb, gene_emb=h_updated, ctx_vec=ctx_vec)

        # Adaptive Auto-Fusion with Symmetrical Imputation and Vector Alignment
        # Vector alignment of isolated genes in parent pathways
        h_raw_aligned = self.isolated_aligner(h)
        # Ensure all inputs are full without zero-vectors
        h_ppi_stream = has_pathway_mask * h_updated + (1. - has_pathway_mask) * h_raw_aligned
        h_refine_stream = has_pathway_mask * refined_gene_emb + (1. - has_pathway_mask) * h_raw_aligned
        h_path_stream = has_pathway_mask * h_pathway_context + (1. - has_pathway_mask) * h_raw_aligned
        gene_emb, auto_loss = self.mux_fuser(h_ppi_stream, h_refine_stream, h_path_stream)
        gene_emb = self.norm_p2g(gene_emb)

        # Final projection to out_dim
        final_gene_emb = self.gene_output_proj(gene_emb) # gene_dim, out_dim
        final_pathway_emb = self.pathway_output_proj(pathway_emb) # pathway_dim, out_dim

        # Prepare attention dictionary
        p2g_attn_dict = {
            'src': pathway_indices.repeat_interleave(batched_ppi.batch_num_nodes()).cpu(), 'dst': global_nids.cpu(), 'weights': alpha_mux.detach().cpu()
        }
        g2p_attn_dict = {
            'src': global_nids.cpu(), 'dst': pathway_indices.repeat_interleave(batched_ppi.batch_num_nodes()).cpu(), 'weights': alpha_readout.detach().cpu()
        }

        return {
            "gene_emb": final_gene_emb,
            "pathway_emb": final_pathway_emb,
            "context_vec": ctx_vec,
            "auto_loss": auto_loss,
            "g2p_attn_dict": g2p_attn_dict,
            "p2g_attn_dict": p2g_attn_dict
        }


    def forward_hybrid(self, g, feat, ctx):
        device = feat.device

        # 1. Context gating
        ctx_vec = self.get_context_vector(ctx) # 1, hidden_dim
        gate = self.context_gate(ctx_vec)

        h = self.gene_input_proj(feat)
        h = self.gene_input_norm(h)
        h = h * gate

        # Get the batched PPI subgraph                                            
        batched_ppi = g.graph_data['batched_ppi_graph'].to(device)                
        # IDs of leaf pathways                                                    
        pathway_indices = g.graph_data['pathway_indices'].to(device) 
        pathway_feat = g.nodes['pathway'].data['feat']

        # Map global gene features to the nodes in the batched subgraph           
        global_nids = batched_ppi.ndata[dgl.NID]                                   
        h_batched = h[global_nids]

        # Identify which genes are processed in the pathways
        has_pathway_mask = torch.zeros(h.size(0), 1, device=device)
        has_pathway_mask.index_fill_(0, global_nids, 1.0)
        
        # Project pathway features
        pathway_query = self.pathway_input_proj(pathway_feat) # pathway_dim, hidden_dim
        pathway_query = self.pathway_input_norm(pathway_query)
        pathway_query = pathway_query * gate

        # Expand pathway query to match every gene in batched PPI
        p_q_batched = pathway_query[pathway_indices].repeat_interleave(batched_ppi.batch_num_nodes(), dim=0)

        # 2. Hybrid message passing (GNN + edge refinement)
        for layer in self.layers:
            # a. Standard GNN layer
            h_batched = layer(batch_ppi, h_batched)
            if self.gnn_type == "gat":
                h_batched = h_batched.view(h_batched.shape[0], -1)

            h_batched = F.relu(h_batched)
            h_batched = self.dropout_layer(h_batched)

        # b. Unified-style edge refinement
        # Assign pathway query to gene
        batched_ppi.ndata['h'] = h_batched
        batched_ppi.ndata['p_q'] = p_q_batched

        def edge_attention(edges):
            # Pathway query for each edge
            attn_in = torch.cat([edges.dst['h'], edges.src['h'], edges.data['p_q']], dim=1)
            attn_in = self.edge_attn_norm(attn_in)
            score = self.rel_attn_a(torch.tanh(self.rel_attn_W(attn_in)))
            return {'score': score}

        batched_ppi.apply_edges(edge_attention)

        # Edge softmax within each pathway's subgrah
        alpha_edge = dgl.ops.edge_softmax(batched_ppi, batched_ppi.edata['score'])

        # Refine features using the edge attention
        batched_ppi.edata['m'] = alpha_edge * batched_ppi.src['h'][batched_ppi.edges()[0]]
        batched_ppi.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_refined'))

        # Residual + Activation
        h_batched = h_batched + batched_ppi.ndata['h_refined']
        h_batched_temp = self.gnn_output_norm(h_batched)

        # Importance scores for every gene instances
        # Since each gene would show up multiple times in different pathways to gather message from genes in the same pathway, attention-based pooling using context is applied here to weight context-related pathway genes
        instance_scores = self.instance_attn(torch.cat([
            h_batched_temp,
            ctx_vec.expand(h_batched.size(0), -1)
        ], dim=-1))
        # Find max score per gene
        gene_max = torch.full((h.size(0), 1), -1e9, device=device)
        gene_max.index_reduce_(0, global_nids, instance_scores, reduce='amax', include_self=False)
        # Segment Softmax handling multi-instance genes
        shifted_scores = instance_scores - gene_max[global_nids]
        exp_scores = torch.exp(shifted_scores)
        # Segment sum
        sum_exp = torch.zeros(h.size(0), 1, device=device).index_add_(0, global_nids, exp_scores)
        instance_weights = exp_scores / (sum_exp[global_nids] + 1e-8)
        # Weighted sum for h_updated
        h_updated = torch.zeros_like(h)
        h_updated.index_add_(0, global_nids, instance_weights * h_batched) # gene_dim, hidden_dim
        h_updated = self.instance_output_norm(h_updated)

        # 3. Mux attention: Pathway -> Gene
        ctx_expand = ctx_vec.expand(h_batched.size(0), -1)
        mux_input = torch.cat([h_batched_temp, ctx_expand], dim=1)
        mux_input = self.mux_input_norm(mux_input)
        mux_scores = self.mux_attn_a(torch.tanh(self.mux_attn_W(mux_input))).squeeze(-1)

        alpha_mux = dgl.softmax_nodes(batched_ppi, mux_scores)

        refined_gene_emb = torch.zeros_like(h)
        weighted_h = h_batched * alpha_mux.unsqueeze(-1)
        refined_gene_emb.index_add_(0, global_nids, weighted_h)
        refined_gene_emb = self.refined_gene_norm(refined_gene_emb)

        # 4. Readout: Gene -> Pathway
        h_pull = h_batched_temp.clone()

        readout_input = torch.cat([h_pull, ctx_expand], dim=1)
        readout_input = self.readout_input_norm(readout_input)
        readout_scores = self.readout_a(torch.tanh(self.readout_W(readout_input))).squeeze(-1)

        alpha_readout = dgl.softmax_nodes(batched_ppi, readout_scores)
        batched_ppi.ndata['weighted_final'] = h_batched * alpha_readout.unsqueeze(-1)
        pathway_emb_batch = dgl.sum_nodes(batched_ppi, 'weighted_final')
        pathway_emb_batch = self.readout_output_norm(pathway_emb_batch)

        # Scatter back to global pathway tensor
        pathway_feat = g.nodes['pathway'].data['feat']
        pathway_emb = self.pathway_input_proj(pathway_feat)
        pathway_emb = self.pathway_input_norm(pathway_emb)
        pathway_emb = pathway_emb * gate

        # Add gene contribution to the leaf pathways
        pathway_emb.index_add_(0, pathway_indices, pathway_emb_batch)
        pathway_emb = self.g2p_norm(pathway_emb)

        # Hierarchy refinement
        pathway_emb, attn_data = self.hierarchy_layer(g, pathway_emb, ctx_vec)

        h_pathway_context, alpha_p2g = self.pathway_attn(g, pathway_emb=pathway_emb, gene_emb=h_updated, ctx_vec=ctx_vec)

        # Adaptive Auto-Fusion with Symmetrical Imputation and Vector Alignment
        # Vector alignment of isolated genes in parent pathways
        h_raw_aligned = self.isolated_aligner(h)
        # Ensure all inputs are full without zero-vectors
        h_ppi_stream = has_pathway_mask * h_updated + (1. - has_pathway_mask) * h_raw_aligned
        h_refine_stream = has_pathway_mask * refined_gene_emb + (1. - has_pathway_mask) * h_raw_aligned
        h_path_stream = has_pathway_mask * h_pathway_context + (1. - has_pathway_mask) * h_raw_aligned
        gene_emb, auto_loss = self.mux_fuser(h_ppi_stream, h_refine_stream, h_path_stream)
        gene_emb = self.norm_p2g(gene_emb)

        # Final projection to out_dim
        final_gene_emb = self.gene_output_proj(gene_emb)
        final_pathway_emb = self.pathway_output_proj(pathway_emb)

        # Prepare attention dictionary
        p2g_attn_dict = {
            'src': pathway_indices.repeat_interleave(batched_ppi.batch_num_nodes()).cpu(), 'dst': global_nids.cpu(), 'weights': alpha_mux.detach().cpu()
        }
        g2p_attn_dict = {
            'src': global_nids.cpu(), 'dst': pathway_indices.repeat_interleave(batched_ppi.batch_num_nodes()).cpu(), 'weights': alpha_readout.detach().cpu()
        }

        return {
            "gene_emb": final_gene_emb,
            "pathway_emb": final_pathway_emb,
            "context_vec": ctx_vec,
            "auto_loss": auto_loss,
            "g2p_attn_dict": g2p_attn_dict,
            "p2g_attn_dict": p2g_attn_dict
        }


    def forward(self, g, feat, ctx):
        if self.mode == "baseline":
            return self.forward_baseline(g, feat, ctx)
        elif self.mode == "unified":
            return self.forward_unified(g, feat, ctx)
        elif self.mode == "mux":
            return self.forward_mux(g, feat, ctx)
        elif self.mode == "hybrid":
            return self.forward_hybrid(g, feat, ctx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class PathwayHierarchyLayer(nn.Module):
    def __init__(self, dim, attn_dim=64):
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        # GMU and MSG
        self.gmu_gate_up = nn.Linear(self.dim * 3, 1)
        self.gmu_gate_down = nn.Linear(self.dim * 3, 1)
        self.msg_encoder_up = nn.Linear(self.dim, self.dim)
        self.msg_encoder_down = nn.Linear(self.dim, self.dim)

        self.W = nn.Linear(self.dim, self.dim)
        # attention for child -> parent
        self.attn_W_up = nn.Linear(3 * self.dim, self.attn_dim)
        self.attn_a_up = nn.Linear(self.attn_dim, 1)
        # attention for parent -> child
        self.attn_W_down = nn.Linear(3 * self.dim, self.attn_dim)
        self.attn_a_down = nn.Linear(self.attn_dim, 1)
        # LayerNorm
        self.attn_input_up_norm = nn.LayerNorm(3 * self.dim)
        self.attn_input_down_norm = nn.LayerNorm(3 * self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, g, pathway_emb, ctx_vec):
        device = pathway_emb.device

        h = pathway_emb.clone()

        # g.levels = [Deepest_leaves, ..., Level_1, Roots]
        # Upward pass leaf to root
        for i in range(len(g.levels) - 1):
            current_level_nodes = g.levels[i].to(device)

            # 1. Identify only the Upward edges from this level to the next level up
            src, dst = g.out_edges(current_level_nodes, etype='child_of')
            if len(src) == 0:
                continue

            # 2. Calculate attention based on current child and current parent
            attn_input_up = torch.cat([h[src], h[dst], ctx_vec.expand(src.size(0), -1)], dim=1)
            attn_input_up = self.attn_input_up_norm(attn_input_up)
            scores_up = self.attn_a_up(torch.tanh(self.attn_W_up(attn_input_up))).squeeze(-1)
            alpha_up = dgl.ops.edge_softmax(g['child_of'], scores_up)

            # GMU gating: decision to let child info through, Arevalo et al. 2017
            gate_up = torch.sigmoid(self.gmu_gate_up(attn_input_up))

            # Synergy-maximizing update, Shankar et al. 2021
            child_msg = torch.tanh(self.msg_encoder_up(h[src]))

            # 3. Parent at level i+1 is informed and ready to be a src in the next iteration
            #h.index_add_(0, dst, alpha_up.unsqueeze(-1) * h[src])
            gated_msg = alpha_up.unsqueeze(-1) * gate_up * child_msg
            h.index_add_(0, dst, gated_msg)

        # Downward pass root to leaf
        for i in range(len(g.levels) - 1, 0, -1):
            current_parent_level_nodes = g.levels[i].to(device)

            # 1. Identify edges going from parent (L_i) to child (L_{i-1})
            src_p, dst_c = g.out_edges(current_parent_level_nodes, etype='parent_of')
            if len(src_p) == 0:
                continue

            # 2. Calculate downward attention
            attn_input_down = torch.cat([h[src_p], h[dst_c], ctx_vec.expand(src_p.size(0), -1)], dim=1)
            attn_input_down = self.attn_input_down_norm(attn_input_down)
            scores_down = self.attn_a_down(torch.tanh(self.attn_W_down(attn_input_down))).squeeze(-1)
            alpha_down = dgl.ops.edge_softmax(g['parent_of'], scores_down)

            # GMU gating
            gate_down = torch.sigmoid(self.gmu_gate_down(attn_input_down))

            parent_msg = torch.tanh(self.msg_encoder_down(h[src_p]))

            # 3. Update child (dst_c) with info from the parent (src_p)
            #h.index_add_(0, dst_c, alpha_down.unsqueeze(-1) * h[src_p])
            gated_msg = alpha_down.unsqueeze(-1) * gate_down * parent_msg
            h.index_add_(0, dst_c, gated_msg)

        h_new = self.norm(self.W(h))

        return h_new, {
            "alpha_up": alpha_up,
            "alpha_down": alpha_down
        }

class PathwayAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, max_rel_depth=15):
        super().__init__()
        # Using DGL's broadcasting
        self.W_p = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.attn_a = nn.Linear(hidden_dim, 1)

        # Input norm: to stabilize hp, hg, and hc before tanh inside UDF
        self.input_norm = nn.LayerNorm(hidden_dim)
        # Output norm: to stabilize the gene_out after fn.sum()
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Hierarchy biases
        self.dist_bias = nn.Embedding(max_rel_depth, 1)
        self.type_bias = nn.Embedding(3, 1)

        # Initialize biases
        with torch.no_grad():
            initial_alpha = 0.4
            for i in range(max_rel_depth):
                penalty = -(math.exp(initial_alpha * i) - 1.0)
                self.dist_bias.weight[i] = penalty

            self.type_bias.weight[0] = 0.0 # no penalty
            self.type_bias.weight[1] = -0.5 # small penalty
            self.type_bias.weight[2] = -2.0 # moderate penalty

    def forward(self, g, pathway_emb, gene_emb, ctx_vec):
        # 1. Project features
        hp = self.input_norm(self.W_p(pathway_emb))
        hg = self.input_norm(self.W_g(gene_emb))
        hc = self.input_norm(self.W_c(ctx_vec))

        # Define triplet
        has_gene_type = ('pathway', 'has_gene', 'gene')
        
        # 2. Define UDF inside forward pass
        def refined_attn_udf(edges):
            combined = torch.tanh(edges.src['hp'] + edges.dst['hg'] + hc)
            raw_scores = self.attn_a(combined).squeeze(-1)

            # Pull hierarchy metadata
            d_bias = self.dist_bias(edges.data['rel_depth']).squeeze(-1)
            t_bias = self.type_bias(edges.data['leaf_type']).squeeze(-1)

            return {'score': raw_scores + d_bias + t_bias}

        with g.local_scope():
            # We store it on the edge temporarily for the UDF
            g.nodes['pathway'].data['hp'] = hp
            g.nodes['gene'].data['hg'] = hg

            # Apply logic without ever writing ctx to the edge data
            g.apply_edges(refined_attn_udf, etype=has_gene_type)

            # Softmax and Aggregation
            alpha = dgl.ops.edge_softmax(
                g[has_gene_type], 
                g.edges[has_gene_type].data['score']
            )
            g.edges[has_gene_type].data['a'] = alpha

            g.update_all(
                fn.u_mul_e('hp', 'a', 'm'), 
                fn.sum('m', 'out'), 
                etype=has_gene_type
            )

            # Capture the output feature so it survives the scope
            gene_out = self.output_norm(g.nodes['gene'].data['out'])
            
            edge_a = g.edges[has_gene_type].data['a']

        # Sandbox is deleted, edge_out and edge_a still exists in memory

        return gene_out, edge_a

class MultiPathwayAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, max_dist=15):
        super().__init__()
        # Output norm: to stabilize the gene_out after fn.sum()
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learn the speficif score offset for each distance and type
        self.dist_bias = nn.Embedding(max_dist+1, 1)
        self.type_bias = nn.Embedding(3, 1) # real leaf, local leaf, parent

        # Embedding dimension
        self.sqrt_dk = math.sqrt(hidden_dim)

        # Initiate biases
        with torch.no_grad():
            initial_alpha = 0.4
            for i in range(max_dist+1):
                penalty = -(math.exp(initial_alpha * i) - 1.0)
                self.dist_bias.weight[i] = penalty

            self.type_bias.weight[0] = 0.0
            self.type_bias.weight[1] = -0.5
            self.type_bias.weight[2] = -2.0

    def forward(self, g, pathway_emb, gene_emb, ctx_vec):
        """
        Ref: Zhang et al. (2022) Transform-based multimodal information fusion for facial expression analysis
        """
        # 1. Get edge endpoints
        src, dst = g.edges(etype='has_gene')

        # 2. Extract the metadata from the edge
        dists = g['has_gene'].data['rel_depth']
        types = g['has_gene'].data['leaf_type']

        # 3. Compute dot product
        Q = self.query(gene_emb)
        K = self.key(pathway_emb)
        V = self.value(pathway_emb)
        hc = self.ctx_proj(ctx_vec)
        
        has_gene_type = ('pathway', 'has_gene', 'gene')

        def Anchor_udf(edges):
            # Dot product (Q * K) / sqrt(d_k)
            dot_score = torch.sum(edges.dst['q'] * edges.src['k'], dim=-1) / self.sqrt_dk

            # Additive Bio-Priors
            d_bias = self.dist_bias(edges.data['rel_depth']).squeeze(-1)
            t_bias = self.type_bias(edges.data['leaf_type']).squeeze(-1)

            return {'score': dot_score + d_bias + t_bias + edges.dst['ctx_match']}

        # 5. Edge softmax for normalization attention
        with g.local_scope():
            # Store temporary tensors for the UDF
            g.nodes['pathway'].data['k'] = K
            g.nodes['pathway'].data['hv'] = V
            g.nodes['gene'].data['q'] = Q
            g.nodes['gene'].data['ctx_match'] = torch.tanh(torch.sum(q * hc, dim=-1)/self.sqrt_dk)

            g.apply_edges(Anchor_udf, etype=has_gene_type)
        
            alpha = dgl.ops.edge_softmax(
                g[has_gene_type], 
                g.edges[has_gene_type].data['score']
            )
            g.edges[has_gene_type].data['a'] = alpha

            # FInal message passing
            g.update_all(
                fn.u_mul_e('hv', 'a', 'm'),
                fn.sum('m', 'h_p'),
                etype=has_gene_type
            )
            gene_out = self.output_norm(g.nodes['gene'].data['h_p'])

        return gene_out, alpha        

class UnifiedPathwayAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attn_dim, max_dist=15):
        super().__init__()
        # Scorer components
        self.attn_W = nn.Linear(hidden_dim * 2, attn_dim)
        self.attn_a = nn.Linear(attn_dim, 1)

        # Manual biases
        self.dist_bias = nn.Embedding(max_dist + 1, 1)
        self.type_bias = nn.Embedding(3, 1)

        # Stability norms
        self.attn_input_norm = nn.LayerNorm(hidden_dim * 2)
        self.p2g_norm = nn.LayerNorm(hidden_dim)

        # Initialize biases
        self._init_biases(max_dist)

        # Learnable scaling factor
        self.bias_scale = nn.Parameter(torch.ones(1) * 0.1) # start with 10% strength

    def _init_biases(self, max_dist):
        with torch.no_grad():
            initial_alpha = 0.4
            for i in range(max_dist + 1):
                penalty = -(math.exp(initial_alpha * i) - 1.0)
                self.dist_bias.weight[i] = penalty

            self.type_bias.weight[0] = 0.0
            self.type_bias.weight[1] = -0.5
            self.type_bias.weight[2] = -2.0

    def forward(self, g, pathway_emb, gene_emb, ctx_vec):
        device = gene_emb.device

        has_gene_type = ('pathway', 'has_gene', 'gene')
        src_p, dst_g = g.edges(etype=has_gene_type)

        # Infer correct device (GPU)
        src_p = src_p.to(device)
        dst_g = dst_g.to(device)

        p_info_per_edge = pathway_emb[src_p]

        ctx_expand = ctx_vec.expand(p_info_per_edge.size(0), -1)

        attn_in = torch.cat([p_info_per_gene, ctx_expand], dim=-1)
        attn_in = self.attn_input_norm(attn_in)

        p2g_scores = self.attn_a(torch.tanh(self.attn_W(attn_in))).squeeze(-1)

        # Calculate manual bias score
        edge_dists = g[has_gene_type].data['rel_depth'].to(device)
        edge_types = g[has_gene_type].data['leaf_type'].to(device)

        d_bias = self.dist_bias(edge_dists).squeeze(-1)
        t_bias = self.type_bias(edge_types).squeeze(-1)

        final_scores = p2g_scores + self.bias_scale * (d_bias + t_bias)

        with g.local_scope():
            alpha = dgl.ops.edge_softmax(g[has_gene_type], final_scores, norm_by='dst')

            gene_from_pathway =  torch.zeros_like(gene_emb)
            gene_from_pathway.index_add_(0, dst_g, alpha.unsqueeze(-1) * p_info_per_edge)

            gene_out = self.p2g_norm(gene_from_pathway)

        return gene_out, alpha


class AutoFusionModule(nn.Module):
    def __init__(self, hidden_dim, num_streams=2, bottleneck_dim=None):
        super(AutoFusionModule, self).__init__()
        # Concatenate GNN and Pathway features for gene
        self.input_dim = num_streams * hidden_dim
        # Fused dim
        self.fused_dim = bottleneck_dim if bottleneck_dim else hidden_dim

        # Hidden layer scaled with the number of inputs
        self.hidden_mid = self.input_dim

        # Transformation Encoder: compress information into z_tm
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_mid),
            nn.LayerNorm(self.hidden_mid),
            nn.ReLU(),
            nn.Linear(self.hidden_mid, self.fused_dim)
        )

        # Reconstruction Decoder: reconstruct z_km from z_tm
        self.decoder = nn.Sequential(
            nn.Linear(self.fused_dim, self.input_dim)
        )

    def forward(self, *streams):
        """
        Ref: Sahu & Vechtomova (2021) Adaptive fusion techniques for multimodal data
            *streams: variable number of [N, hidden_dim] tensors
        returns:
            - z_tm: fused joint representation
            - reconstruction_loss: MSE loss J_tr = ||z_hat - z_km||^2
        """
        # Validation: if num_streams is equal to length of streams
        if len(streams) != (self.input_dim // streams[0].size(-1)):
            raise ValueError(f"Expected {self.input_dim // streams[0].size(-1)} streams,"
                            f"but got {len(streams)}")

        # 1. Preliminary concatenation: z_km
        z_km = torch.cat(streams, dim=-1)

        # 2. Adaptive transformation to latent vector: z_tm
        z_tm = self.encoder(z_km)

        # 3. Information preservation
        z_hat = self.decoder(z_tm)

        # 4. Calculate Euclidean distance
        reconstruction_loss = F.mse_loss(z_hat, z_km)

        return z_tm, reconstruction_loss


class MHHGNN(nn.Module):
    """
    Multiplex Heterogeneous Hierarchical GNN for gene-pathway networks
    """
    def __init__(
        self,
        context_info, # the context of GEO data
        gnn_type: str = "gcn",
        num_layers: int = 2,
        in_dim: int = 128,
        hidden_dim: int = 512,
        emb_dim: int = 128, # out_dim
        attn_dim: int = 64,
        dropout_prob: float = 0.2,
        device='cpu'
    ):
        super().__init__()
        # Save hyperparameters
        self.device = device
        self.emb_dim = emb_dim

        # Gene encoder with PPI message passing + attention
        self.encoder = GeneEncoder(
            context_info=context_info,
            gnn_type=gnn_type.lower(),
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=emb_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        # loss
        self.loss_fn = PathwayNegativeSamplingLoss(device=device)


    def forward(self, g, ctx, sampler=None, compute_loss=False):
        """
        Forward pass:
        - g: PPI reactome pathway graph (DGLGraph)
        - feat: gene features (num_genes, in_dim)
        - pathway_nodes: dict {pathway_id: [gene_ids]}
        - gene2pathways: dict {gene_id: [pathway_ids]}
        """
        gene_feat = g.nodes['gene'].data['feat']
        # Encode genes and pathways
        enc_out = self.encoder.forward(g, gene_feat, ctx)

        gene_emb = enc_out["gene_emb"]
        pathway_emb = enc_out["pathway_emb"]
        # Pack embeddings for loss
        embeddings = {
            "gene": gene_emb,
            "pathway": pathway_emb
        }
        # If no loss -> inference mode
        if not compute_loss:
            return enc_out

        # Sampling
        sample_dict = sampler.sample_pos_neg(ctx)
        # Loss
        loss = self.loss_fn(embeddings, sample_dict)
        # For now, just return gene embeddings and attention maps
        return loss, enc_out

