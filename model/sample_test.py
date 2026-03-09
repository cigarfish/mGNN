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
        g,
        num_pathways,
        max_reach_dist=2,
        max_motifs_per_pathway=20
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
        self.g = g
        self.num_pathways = num_pathways
        self.max_reach_dist = max_reach_dist
        self.max_motifs_per_pathway = max_motifs_per_pathway
        
        # Build neighbors only within particular pathway
        self.pathway_gene_neighbors = defaultdict(lambda: defaultdict(list))
        self.gene2direct_pathways = defaultdict(set)

        pathway_mask = g.edges['ppi'].data['pathway_mask']
        take_mask = g.nodes['pathway'].data['take_mask']
        src, dst = g.edges(etype='ppi')

        for eid in range(g.num_edges('ppi')):
            s = src[eid].item()
            d = dst[eid].item()
            # pathways this edge belongs to
            pws = torch.where(pathway_mask[eid])[0]
            for pw in pws:
                pw = pw.item()
                # skip pathways not taken
                if not take_mask[pw]:
                    continue

                self.pathway_gene_neighbors[pw][s].append(d)
                self.pathway_gene_neighbors[pw][d].append(s)

                self.gene2direct_pathways[s].add(pw)
                self.gene2direct_pathways[d].add(pw)

        # pathway genes in string PPI
        self.pathway_ppi_genes = {
                p: list(neigh.keys())
                for p, neigh in self.pathway_gene_neighbors.items()
        }
        # Precompute valid pathway pool
        #self.valid_pathways = torch.where(take_mask)[0]

    # --------------------------------------------------
    # Core logic
    # --------------------------------------------------
    def _genes_in_pathway(self, p):
        return self.g.predecessors(p, etype='in_pathway').tolist()

    def _gene_neighbors(self, gene_id, pathway_id):
        return self.pathway_gene_neighbors[pathway_id].get(gene_id, [])

    def _sample_motif(self, pathway_id, start_gene=None):
        #genes = self._genes_in_pathway(pathway_id)
        #genes = list(self.pathway_gene_neighbors[pathway_id].keys())
        genes = self.pathway_ppi_genes.get(pathway_id, [])

        if len(genes) < 3:
            return None

        if start_gene is not None:
            if start_gene not in self.pathway_gene_neighbors[pathway_id]:
                return None
            g0 = start_gene
        else:
            g0 = random.choice(genes)

        nbr1 = [g for g in self._gene_neighbors(g0, pathway_id) if g != g0]
        if not nbr1:
            return None

        g1 = random.choice(nbr1)

        nbr2 = [g for g in self._gene_neighbors(g1, pathway_id) if g!= g0 and g!= g1]
        if not nbr2:
            return None

        g2 = random.choice(nbr2)

        return (pathway_id, g0, g1, g2)

    def _sample_pathway_jump(self, gene_id, current_pathway):
        # pathways containing this gene
        pathways = self.gene2direct_pathways.get(gene_id, set())

        candidates = [
                pw for pw in pathways
                if pw != current_pathway #and self.take_mask[pw]
        ]

        if not candidates:
            return None

        return random.choice(candidates)

    def _sample_pathways_jump(self, gene_id, current_pathway, k=3):
        pathways = self.gene2direct_pathways.get(gene_id, set())

        candidates = [
                pw for pw in pathways
                if pw != current_pathway
        ]

        if not candidates:
            return []

        if len(candidates) <= k:
            return candidates

        return random.sample(candidates, k)

    def _positive_sample(self, ctx, max_genes_per_pathway=20):
        gg_pairs = set()
        gp_pairs = set()
        pg_pairs = set()

        for p1 in random.sample(list(ctx.pathways), len(ctx.pathways)):

            sampled_genes = set()

            trials = 0
            max_trials = 100
            motif_count = 0

            while (
                len(sampled_genes) < max_genes_per_pathway 
                and trials < max_trials
                and motif_count < self.max_motifs_per_pathway
            ):

                trials += 1

                motif = self._sample_motif(p1, start_gene=None)
                if motif is None:
                    continue

                _, g0, g1, g2 = motif

                sampled_genes.update([g0, g1, g2])

                # gene-gene
                gg_pairs.add(tuple(sorted((g0, g1))))
                gg_pairs.add(tuple(sorted((g1, g2))))

                # gene-pathway
                gp_pairs.update([
                    (g0, p1),
                    (g1, p1),
                    (g2, p1)
                ])

                # pathway-gene
                pg_pairs.update([
                    (p1, g0),
                    (p1, g1),
                    (p1, g2)
                ])

                # pathway jump
                p2 = self._sample_pathway_jump(g2, p1)

                if p2 is None:
                    continue

                motif2 = self._sample_motif(p2, start_gene=g2)

                if motif2 is None:
                    continue

                _, g2, g3, g4 = motif2

                gg_pairs.add(tuple(sorted((g2, g3))))
                gg_pairs.add(tuple(sorted((g3, g4))))

                gp_pairs.update([
                    (g2, p2),
                    (g3, p2),
                    (g4, p2)
                ])

                pg_pairs.update([
                    (p2, g2),
                    (p2, g3),
                    (p2, g4)
                ])

                motif_count += 1

            for g in sampled_genes:
                jump_pathways = self._sample_pathways_jump(g, p1, k=3)

                for p2 in jump_pathways:
                    gp_pairs.add((g, p2))
                    pg_pairs.add((p2, g))

        return {
            "gg": list(gg_pairs),
            "gp": list(gp_pairs),
            "pg": list(pg_pairs)
        }


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

