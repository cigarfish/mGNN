#!/usr/bin/env python3

import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import defaultdict

import dgl
from dgl.dataloading import Collator, BlockSampler
# noinspection PyProtectedMember

class PathwayNegativeSampler:
    """
    Global sampler that knows biological reachability
    and valid negative constraints.
    """
    def __init__(
        self,
        g,
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
        self.max_reach_dist = max_reach_dist
        self.max_motifs_per_pathway = max_motifs_per_pathway
        
        # Build neighbors only within particular pathway
        self.pathway_gene_neighbors = defaultdict(lambda: defaultdict(list))
        self.gene2direct_pathways = defaultdict(set)

        pathway_mask = g.edges['ppi'].data['pathway_mask']
        take_mask = g.nodes['pathway'].data['take_mask']
        src, dst = g.edges(etype='ppi')

        self.ppi_edges = set()
        for eid in range(g.num_edges('ppi')):
            s = src[eid].item()
            d = dst[eid].item()
            self.ppi_edges.add(tuple(sorted((s,d))))
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
        # degree of each gene
        self.gene_degree = defaultdict(int)
        for g1, g2 in zip(src.tolist(), dst.tolist()):
            edge = tuple(sorted((g1, g2)))
            self.gene_degree[g1] += 1
            self.gene_degree[g2] += 1
        self.all_genes = list(range(self.g.num_nodes('gene')))
        
        # build degree bins
        self.degree_bins = defaultdict(list)
        for g0, deg in self.gene_degree.items():
            bin_id = int(math.log2(deg + 1))
            self.degree_bins[bin_id].append(g0)

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

    def sample_pos_neg(self, ctx, max_genes_per_pathway=20, neg_multiplier=5):
        # ----------------
        # Positive samples
        # ----------------
        gg_pairs = set()
        gp_pairs = set()
        pg_pairs = set()

        for p1, pdata in random.sample(list(ctx.pathway_genes.items()), len(ctx.pathway_genes)):

            enriched_genes = list(pdata["genes"])
            pathway_genes = self.pathway_ppi_genes.get(p1, [])
            non_enriched_genes = [g for g in pathway_genes if g not in enriched_genes]

            if len(enriched_genes) == 0:
                continue

            trials = 0
            max_trials = 100
            motif_count = 0
            sampled_genes = set()

            while (
                len(sampled_genes) < max_genes_per_pathway 
                and trials < max_trials
                and motif_count < self.max_motifs_per_pathway
            ):

                trials += 1

                # prioritize enriched genes
                if random.random() < 0.8 or not non_enriched_genes:
                    g0 = random.choice(enriched_genes)
                else:
                    g0 = random.choice(non_enriched_genes)

                nbr1 = [g for g in self._gene_neighbors(g0, p1) if g != g0]

                if not nbr1:
                    continue

                g1 = random.choice(nbr1)

                nbr2 = [g for g in self._gene_neighbors(g1, p1) if g != g0 and g != g1]

                if not nbr2:
                    continue

                g2 = random.choice(nbr2)

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

                nbr3 = [g for g in self._gene_neighbors(g2, p2) if g != g2]

                if not nbr3:
                    continue

                g3 = random.choice(nbr3)

                nbr4 = [g for g in self._gene_neighbors(g3, p2) if g != g2 and g != g3]

                if not nbr4:
                    continue

                g4 = random.choice(nbr4)

                motif2 = self._sample_motif(p2, start_gene=g2)

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

        # ---------------
        # Negative sample
        # ---------------
        # for gene-gene pairs
        pos_gg_count = len(gg_pairs)
        neg_gg = set()

        for g1, g2 in gg_pairs:

            anchor = g1 if random.random() < 0.5 else g2

            deg = self.gene_degree.get(anchor, 1)
            bin_id = int(math.log2(deg + 1))
            candidates = self.degree_bins.get(bin_id)
            if not candidates:
                candidates = (
                    self.degree_bins.get(bin_id-1)
                    or self.degree_bins.get(bin_id+1)
                    or self.all_genes
                )

            sampled = 0
            trials = 0
            max_trials = 100

            anchor_pathways = self.gene2direct_pathways.get(anchor, set())

            while sampled < 2 and trials < max_trials:
                trials += 1
                g_neg = random.choice(candidates)

                if g_neg == anchor:
                    continue

                edge = tuple(sorted((anchor, g_neg)))

                # skip real PPI edges
                if edge in self.ppi_edges:
                    continue

                # avoid same pathway genes
                if len(anchor_pathways.intersection(self.gene2direct_pathways.get(g_neg, set()))) > 0:
                    continue

                neg_gg.add(edge)
                sampled += 1

        # for gene-pathway pairs
        pos_gp_count = len(gp_pairs)
        neg_target = pos_gp_count * neg_multiplier
        neg_gp = set()
        neg_pg = set()
        num_pathways = self.g.num_nodes('pathway')
        all_pathways = list(range(num_pathways))

        while len(neg_gp) < neg_target:
            g = random.choice(list(self.gene2direct_pathways.keys()))
            p = random.choice(all_pathways)
            # skip if gene already belongs to pathway or pathway not TAKE
            if p in self.gene2direct_pathways[g]:
                continue
            if not self.g.nodes['pathway'].data['take_mask'][p]:
                continue

            neg_gp.add((g, p))
            neg_pg.add((p, g))

        return {
            "gg_pos": list(gg_pairs),
            "gp_pos": list(gp_pairs),
            "pg_pos": list(pg_pairs),
            "gg_neg": list(neg_gg),
            "gp_neg": list(neg_gp),
            "pg_neg": list(neg_pg)
        }

