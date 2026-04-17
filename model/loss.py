import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayNegativeSamplingLoss(nn.Module):

    def __init__(self, lambda_auto=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.lambda_auto = lambda_auto

    def dot_score(self, a, b, ctx_weight):
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)

        # DistMult: (A * W * B).sum()
        return (a * ctx_weight * b).sum(dim=1)

    def forward(self, embeddings, sample_dict, context_vec, auto_loss):

        gene_emb = embeddings['gene']
        pathway_emb = embeddings['pathway']

        # ------------------
        # gene-pathway positive
        # ------------------
        gp_pos = torch.tensor(sample_dict['gp_pos'], dtype=torch.long, device=self.device)
        g_idx, p_idx = gp_pos[:,0], gp_pos[:,1]

        gp_pos_scores = self.dot_score(gene_emb[g_idx], pathway_emb[p_idx], context_vec)
        gp_pos_loss = F.binary_cross_entropy_with_logits(
            gp_pos_scores, torch.ones_like(gp_pos_scores)
        )

        # ------------------
        # gene-pathway negative
        # ------------------
        gp_neg = torch.tensor(sample_dict['gp_neg'], dtype=torch.long, device=self.device)
        g_idx, p_idx = gp_neg[:,0], gp_neg[:,1]

        gp_neg_scores = self.dot_score(gene_emb[g_idx], pathway_emb[p_idx], context_vec)
        gp_neg_loss = F.binary_cross_entropy_with_logits(
            gp_neg_scores, torch.zeros_like(gp_neg_scores)
        )

        L_gp = gp_pos_loss + gp_neg_loss

        # ------------------
        # gene-gene positive
        # ------------------
        gg_pos = torch.tensor(sample_dict['gg_pos'], dtype=torch.long, device=self.device)
        g1, g2 = gg_pos[:,0], gg_pos[:,1]

        gg_pos_scores = self.dot_score(gene_emb[g1], gene_emb[g2], context_vec)
        gg_pos_loss = F.binary_cross_entropy_with_logits(
            gg_pos_scores, torch.ones_like(gg_pos_scores)
        )

        # ------------------
        # gene-gene negative
        # ------------------
        gg_neg = torch.tensor(sample_dict['gg_neg'], dtype=torch.long, device=self.device)
        g1, g2 = gg_neg[:,0], gg_neg[:,1]

        gg_neg_scores = self.dot_score(gene_emb[g1], gene_emb[g2], context_vec)
        gg_neg_loss = F.binary_cross_entropy_with_logits(
            gg_neg_scores, torch.zeros_like(gg_neg_scores)
        )

        L_gg = gg_pos_loss + gg_neg_loss

        # ------------------
        # final loss
        # ------------------
        loss = L_gp + 0.5 * L_gg + self.lambda_auto * auto_loss

        return loss

