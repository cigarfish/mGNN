#!/usr/bin/env python3

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from dgl.dataloading import Collator, BlockSampler
# noinspection PyProtectedMember
from dgl.dataloading.pytorch import _pop_blocks_storage, _restore_blocks_storage

class PathwayContext:
    """
    Represents one GEO dataset (or one biological condition).
    """
    def __init__(self, pathway_ids):
        """
        Parameters
        ----------
        pathway_ids : Iterable[int]
            Enriched pathway node IDs for this GEO dataset
        """
        self.pathways = set(pathway_ids)

    def contains(self, pathway_id: int) -> bool:
        return pathway_id in self.pathways

class GeoBatchContext:
    """
    A batch of GEO datasets.
    """

    def __init__(self, contexts):
        """
        contexts : List[PathwayContext]
        """
        self.contexts = contexts

    def __len__(self):
        return len(self.contexts)

class PathwayNegativeSampler:
    """
    Global sampler that knows biological reachability
    and valid negative constraints.
    """
    def __init__(
        self,
        gene2pathway_dist,
        take_mask,
        num_pathways,
        max_reach_dist=2,
    ):
        """
        Parameters
        ----------
        gene2pathway_dist : Dict[int, Dict[int, int]]
            gene_id -> {pathway_id: min_distance}

        take_mask : BoolTensor [num_pathways]
            Whether a pathway is a valid leaf / learning target

        num_pathways : int
            Total number of pathways in graph

        max_reach_dist : int
            Pathways within this distance are NOT allowed as negatives
        """
        self.gene2pathway_dist = gene2pathway_dist
        self.take_mask = take_mask
        self.num_pathways = num_pathways
        self.max_reach_dist = max_reach_dist

        # Precompute valid pathway pool
        self.valid_pathways = torch.where(take_mask)[0]

    # --------------------------------------------------
    # Core logic
    # --------------------------------------------------

    def reachable_pathways(self, gene_id: int):
        """
        Pathways biologically reachable from gene.
        """
        dist_map = self.gene2pathway_dist.get(gene_id, {})
        return {
            p for p, d in dist_map.items()
            if d <= self.max_reach_dist
        }

    def sample_negatives(
        self,
        gene_ids,
        pathway_context: PathwayContext,
        num_neg:int,
        device,
    ):
        """
        Sample negative pathways for each gene.

        Returns
        -------
        neg_pathways : LongTensor [B, num_neg]
        """
        B = len(gene_ids)
        neg_samples = []

        for g in gene_ids.tolist():
            forbidden = set()

            # 1) GEO-enriched pathways
            forbidden |= pathway_context.pathways

            # 2) Biologically reachable pathways
            forbidden |= self.reachable_pathways(g)

            # 3) Invalid (non-leaf) already filtered by valid_pathways

            candidates = [
                p.item()
                for p in self.valid_pathways
                if p.item() not in forbidden
            ]

            if len(candidates) == 0:
                raise RuntimeError(
                    f"No valid negatives for gene {g}"
                )

            sampled = torch.randint(
                0, len(candidates),
                (num_neg,),
                device=device
            )
            neg_samples.append(
                torch.tensor(
                    [candidates[i] for i in sampled],
                    device=device
                )
            )

        return torch.stack(neg_samples, dim=0)


class PathwayNegativeSamplingLoss(nn.Module):
    def __init__(
        self,
        num_genes: int,
        num_pathways: int,
        embed_dim: int,
        num_neg: int=10
    ):
        super().__init__()

        self.num_genes = num_genes
        self.num_pathways = num_pathways
        self.embed_dim = embed_dim
        self.num_neg = num_neg

        self.gene_weights = nn.Parameter(torch.empty(num_genes, embed_dim))
        self.pathway_weights = nn.Parameter(torch.empty(num_pathways, embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gene_weights, std=1.0 / math.sqrt(self.embed_dim))
        nn.init.normal_(self.pathway_weights, std=1.0 / math.sqrt(self.embed_dim))

    # --------------------------------------------------
    # Sampling helpers
    # --------------------------------------------------
    def _sample_gene_neg(self, B, device):
        return torch.randint(0, self.num_genes, (B, self.num_neg), device=device)

    def _sample_pathway_neg(self, B, device):
        return torch.randint(0, self.num_pathways, (B, self.num_neg), device=device)

    # --------------------------------------------------
    # Atomic losses
    # --------------------------------------------------
    def _gene_gene(self, src, ctx, gene_embeds):
        h = gene_embeds[src]
        w = self.gene_weights[ctx]

        pos = F.logsigmoid(torch.sum(h * w, dim=1))

        neg_ids = self._sample_gene_neg(len(src), h.device)
        neg_w = self.gene_weights[neg_ids]

        neg = F.logsigmoid(
            -torch.bmm(neg_w, h.unsqueeze(-1)).squeeze(-1)
        ).sum(dim=1)

        return -(pos + neg).mean()

    def _gene_pathway(self, genes, pathways, gene_embeds):
        h = gene_embeds[genes]
        w = self.pathway_weights[pathways]

        pos = F.logsigmoid(torch.sum(h * w, dim=1))

        neg_ids = self._sample_pathway_neg(len(genes), h.device)
        neg_w = self.pathway_weights[neg_ids]

        neg = F.logsigmoid(
            -torch.bmm(neg_w, h.unsqueeze(-1)).squeeze(-1)
        ).sum(dim=1)

        return -(pos + neg).mean()

    def _pathway_gene(self, pathways, genes, pathway_embeds):
        h = pathway_embeds[pathways]
        w = self.gene_weights[genes]

        pos = F.logsigmoid(torch.sum(h * w, dim=1))

        neg_ids = self._sample_gene_neg(len(pathways), h.device)
        neg_w = self.gene_weights[neg_ids]

        neg = F.logsigmoid(
            -torch.bmm(neg_w, h.unsqueeze(-1)).squeeze(-1)
        ).sum(dim=1)

        return -(pos + neg).mean()

    # --------------------------------------------------
    #  Unified forward
    # --------------------------------------------------
    def forward(
        self,
        gene_embeds,
        pathway_embeds,
        gene_gene_pairs=None,
        gene_pathway_pairs=None,
        pathway_gene_pairs=None,
        weights=None,
        return_breakdown=False
    ):
        """
        All *_pairs are tuples of ID tensors
        """

        if weights is None:
            weights = {
                "gene_gene": 1.0,
                "gene_pathway": 1.0,
                "pathway_gene": 0.5
            }

        total_loss = 0.0
        breakdown = {}

        if gene_gene_pairs is not None:
            src, ctx = gene_gene_pairs
            L = self._gene_gene(src, ctx, gene_embeds)
            total_loss += weights["gene_gene"] * L
            breakdown["gene_gene"] = L.item()

        if gene_pathway_pairs is not None:
            g, p = gene_pathway_pairs
            L = self._gene_pathway(g, p, gene_embeds)
            total_loss += weights["gene_pathway"] * L
            breakdown["gene_pathway"] = L.item()

        if pathway_gene_pairs is not None:
            p, g = pathway_gene_pairs
            L = self._pathway_gene(p, g, pathway_embeds)
            total_loss += weights["pathway_gene"] * L
            breakdown["pathway_gene"] = L.item()

        if return_breakdown:
            return total_loss, breakdown

        return total_loss



class PathwayNegativeSamplingLossSimple(nn.Module):
    def __init__(self, num_genes, embed_dim, num_neg_samples):
        super().__init__()
        self.num_neg = num_neg_samples
        self.weights = nn.Parameter(
            torch.randn(num_genes, embed_dim) / math.sqrt(embed_dim)
        )

    def forward(self, heads, head_embeds, tails):
        """
        heads: (B,)
        head_embeds: (B, D)
        tails: (B,)
        """
        B, D = head_embeds.shape

        # positive
        pos_w = self.weights[tails]                       # (B, D)
        pos_score = torch.sum(head_embeds * pos_w, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # negative
        neg_tails = torch.randint(
            0, self.weights.size(0),
            (B, self.num_neg),
            device=head_embeds.device
        )
        neg_w = self.weights[neg_tails]                   # (B, K, D)
        neg_score = torch.bmm(
            neg_w.neg(),
            head_embeds.unsqueeze(-1)
        ).squeeze(-1)                                     # (B, K)
        neg_loss = F.logsigmoid(neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

