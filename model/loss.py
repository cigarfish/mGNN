import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayNegativeSamplingLoss(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def dot_score(self, a, b):
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        return (a * b).sum(dim=1)

    def forward(self, embeddings, sample_dict):

        gene_emb = embeddings['gene']
        pathway_emb = embeddings['pathway']

        # ------------------
        # gene-pathway positive
        # ------------------
        gp_pos = torch.tensor(sample_dict['gp_pos'], dtype=torch.long, device=self.device)
        g_idx, p_idx = gp_pos[:,0], gp_pos[:,1]

        pos_scores = self.dot_score(gene_emb[g_idx], pathway_emb[p_idx])
        #gp_pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
        gp_pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        # ------------------
        # gene-pathway negative
        # ------------------
        gp_neg = torch.tensor(sample_dict['gp_neg'], dtype=torch.long, device=self.device)
        g_idx, p_idx = gp_neg[:,0], gp_neg[:,1]

        neg_scores = self.dot_score(gene_emb[g_idx], pathway_emb[p_idx])
        #gp_neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-8).mean()
        gp_neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        L_gp = gp_pos_loss + gp_neg_loss

        # ------------------
        # gene-gene positive
        # ------------------
        gg_pos = torch.tensor(sample_dict['gg_pos'], dtype=torch.long, device=self.device)
        g1, g2 = gg_pos[:,0], gg_pos[:,1]

        gg_scores = self.dot_score(gene_emb[g1], gene_emb[g2])
        #gg_pos_loss = -torch.log(torch.sigmoid(gg_scores) + 1e-8).mean()
        gg_pos_loss = F.binary_cross_entropy_with_logits(
            gg_scores, torch.ones_like(gg_scores)
        )

        # ------------------
        # gene-gene negative
        # ------------------
        gg_neg = torch.tensor(sample_dict['gg_neg'], dtype=torch.long, device=self.device)
        g1, g2 = gg_neg[:,0], gg_neg[:,1]

        gg_neg_scores = self.dot_score(gene_emb[g1], gene_emb[g2])
        #gg_neg_loss = -torch.log(torch.sigmoid(-gg_neg_scores) + 1e-8).mean()
        gg_neg_loss = F.binary_cross_entropy_with_logits(
            gg_neg_scores, torch.zeros_like(gg_neg_scores)
        )

        L_gg = gg_pos_loss + gg_neg_loss

        # ------------------
        # final loss
        # ------------------
        loss = L_gp + 0.5 * L_gg

        return loss

