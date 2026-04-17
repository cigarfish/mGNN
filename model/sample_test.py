#!/usr/bin/env python3

import math
import random
import numpy as np

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

        nbr1 = [gi for gi in self._gene_neighbors(g0, pathway_id) if gi != g0]
        if not nbr1:
            return None

        g1 = random.choice(nbr1)

        nbr2 = [gi for gi in self._gene_neighbors(g1, pathway_id) if gi!= g0 and gi!= g1]
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
            non_enriched_genes = [gi for gi in pathway_genes if gi not in enriched_genes]

            if len(enriched_genes) == 0:
                continue

            trials = 0
            max_trials = 100
            sampled_genes = set()

            while (
                len(sampled_genes) < max_genes_per_pathway 
                and trials < max_trials
            ):

                trials += 1

                # prioritize enriched genes
                if random.random() < 0.8 or not non_enriched_genes:
                    g0 = random.choice(enriched_genes)
                else:
                    g0 = random.choice(non_enriched_genes)

                nbr1 = [gi for gi in self._gene_neighbors(g0, p1) if gi != g0]

                if not nbr1:
                    continue

                g1 = random.choice(nbr1)

                nbr2 = [gi for gi in self._gene_neighbors(g1, p1) if gi != g0 and gi != g1]

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

                nbr3 = [gi for gi in self._gene_neighbors(g2, p2) if gi != g2]

                if not nbr3:
                    continue

                g3 = random.choice(nbr3)

                nbr4 = [gi for gi in self._gene_neighbors(g3, p2) if gi != g2 and gi != g3]

                if not nbr4:
                    continue

                g4 = random.choice(nbr4)

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

            for gi in sampled_genes:
                jump_pathways = self._sample_pathways_jump(gi, p1, k=3)

                for p2 in jump_pathways:
                    gp_pairs.add((gi, p2))
                    pg_pairs.add((p2, gi))

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
            gi = random.choice(list(self.gene2direct_pathways.keys()))
            pi = random.choice(all_pathways)
            # skip if gene already belongs to pathway or pathway not TAKE
            if pi in self.gene2direct_pathways[gi]:
                continue
            if not self.g.nodes['pathway'].data['take_mask'][pi]:
                continue

            neg_gp.add((gi, pi))
            neg_pg.add((pi, gi))

        return {
            "gg_pos": list(gg_pairs),
            "gp_pos": list(gp_pairs),
            "pg_pos": list(pg_pairs),
            "gg_neg": list(neg_gg),
            "gp_neg": list(neg_gp),
            "pg_neg": list(neg_pg)
        }

    def sample_pos_neg_context(self, ctx, hard_refs=None, medium_refs=None, easy_refs=None, max_genes_per_pathway=20, neg_multiplier=5):
        # ----------------
        # Positive samples
        # ----------------
        gg_pairs = set()
        gp_pairs = set()
        pg_pairs = set()

        for p1, pdata in random.sample(list(ctx.pathway_genes.items()), len(ctx.pathway_genes)):

            enriched_genes = list(pdata["genes"])
            pathway_genes = self.pathway_ppi_genes.get(p1, [])
            non_enriched_genes = [gi for gi in pathway_genes if gi not in enriched_genes]

            if len(enriched_genes) == 0:
                continue

            trials = 0
            max_trials = 100
            sampled_genes = set()

            while (
                len(sampled_genes) < max_genes_per_pathway 
                and trials < max_trials
            ):

                trials += 1

                # prioritize enriched genes
                if random.random() < 0.8 or not non_enriched_genes:
                    g0 = random.choice(enriched_genes)
                else:
                    g0 = random.choice(non_enriched_genes)

                nbr1 = [gi for gi in self._gene_neighbors(g0, p1) if gi != g0]

                if not nbr1:
                    continue

                g1 = random.choice(nbr1)

                nbr2 = [gi for gi in self._gene_neighbors(g1, p1) if gi != g0 and gi != g1]

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

                nbr3 = [gi for gi in self._gene_neighbors(g2, p2) if gi != g2]

                if not nbr3:
                    continue

                g3 = random.choice(nbr3)

                nbr4 = [gi for gi in self._gene_neighbors(g3, p2) if gi != g2 and gi != g3]

                if not nbr4:
                    continue

                g4 = random.choice(nbr4)

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

            for gi in sampled_genes:
                jump_pathways = self._sample_pathways_jump(gi, p1, k=3)

                for p2 in jump_pathways:
                    gp_pairs.add((gi, p2))
                    pg_pairs.add((p2, gi))

        # ---------------
        # Negative sample
        # ---------------
        """
        Negative consists of 
        1. Hard contextual (50%): high similarity from different organ
        2. Medium contextual (25%): low similarity from different organ
        3. Easy contextual (15%): low similarity from the same organ
        4. Easy topological (10%): different hub in the String PPI
        """
        pos_gg_count = len(gg_pairs)
        target_topo_neg_gg = int(pos_gg_count * 2 * 0.1)
        target_hard_neg_gg = int(pos_gg_count * 2 * 0.5)
        target_medium_neg_gg = int(pos_gg_count * 2 * 0.25)
        target_easy_neg_gg = int(pos_gg_count * 2 * 0.15)
        
        pos_gp_count = len(gp_pairs)
        target_topo_neg_gp = int(pos_gp_count * neg_multiplier * 0.1)
        target_hard_neg_gp = int(pos_gp_count * neg_multiplier * 0.5)
        target_medium_neg_gp = int(pos_gp_count * neg_multiplier * 0.25)
        target_easy_neg_gp = int(pos_gp_count * neg_multiplier * 0.15)

        hard_neg_gg = set()
        hard_neg_gp = set()
        hard_neg_pg = set()

        hard_neg_gg_temp = set()
        hard_neg_gp_temp = set()
        hard_neg_pg_temp = set()
        # Sample hard negatives
        if hard_refs:
            for ref in hard_refs:
                # Identify hard pathways
                ref_pathway_items = list(ref.pathway_genes.items())
                # Go over all pathways in this hard negative sample
                for p_id, p_data_ref in random.sample(ref_pathway_items, len(ref_pathway_items)):
                    enriched_genes = set(p_data_ref["gene"])
                    if len(enriched_genes) == 0:
                        continue

                    # Existence check
                    if p_id in ctx.pathway_genes:
                        query_p_data = ctx.pathway_genes[p_id]
                        query_p_genes = set(query_p_data["genes"])

                        # Similarity check
                        intersection = len(query_p_genes.intersection(enriched_genes))
                        union = len(query_p_genes.union(enriched_genes))
                        pathway_sim = intersection / union if union > 0 else 0

                        # If the pathway is effectively the same in both, skip it
                        if pathway_sim > 0.6:
                            continue

                    pathway_genes = self.pathway_ppi_genes.get(p_id, [])
                    non_enriched_genes = [gi for gi in pathway_genes if gi not in enriched_genes]

                    trials = 0
                    max_trials = 1000
                    sampled_genes = set()

                    while (
                        len(sampled_genes) < max_genes_per_pathway
                        and trials < max_trials
                    ):
                        trials += 1

                        # prioritize enriched genes
                        if random.random() < 0.8 or not non_enriched_genes:
                            g0 = random.choice(enriched_genes)
                        else:
                            g0 = random.choice(non_enriched_genes)

                        nbr1 = [gi for gi in self._gene_neighbors(g0, p_id) if gi != g0]
                        
                        if not nbr1:
                            continue

                        g1 = random.choice(nbr1)

                        edge01 = tuple(sorted((g0, g1)))

                        if (edge01 in gg_pairs):
                            continue

                        nbr2 = [gi for gi in self._gene_neighbors(g1, p_id) if gi != g0 and gi != g1]

                        if not nbr2:
                            continue

                        g2 = random.choice(nbr2)

                        edge12 = tuple(sorted((g1, g2)))

                        if (edge12 in gg_pairs):
                            continue

                        sampled_genes.update([g0, g1, g2])
                        
                        # gene-gene
                        hard_neg_gg_temp.add(tuple(sorted((g0, g1))))
                        hard_neg_gg_temp.add(tuple(sorted((g1, g2))))

                        # gene-pathway & pathway_gene
                        if (g0, p_id) not in gp_pairs:
                            hard_neg_gp_temp.add((g0, p_id))
                            hard_neg_pg_temp.add((p_id, g0))
                        if (g1, p_id) not in gp_pairs:
                            hard_neg_gp_temp.add((g1, p_id))
                            hard_neg_pg_temp.add((p_id, g1))
                        if (g2, p_id) not in gp_pairs:
                            hard_neg_gp_temp.add((g2, p_id))
                            hard_neg_pg_temp.add((p_id, g2))

                        # pathway jump
                        p2 = self._sample_pathway_jump(g2, p_id)

                        if p2 is None:
                            continue

                        nbr3 = [gi for gi in self._gene_neighbors(g2, p2) if gi != g2]

                        if not nbr3:
                            continue

                        g3 = random.choice(nbr3)

                        edge23 = tuple(sorted((g2, g3)))
                        if edge23 in gg_pairs:
                            continue
                        
                        nbr4 = [gi for gi in self._gene_neighbors(g3, p2) if gi != g2 and gi != g3]

                        if not nbr4:
                            continue

                        g4 = random.choice(nbr4)

                        edge34 = tuple(sorted((g3, g4)))
                        if edge34 in gg_pairs:
                            continue

                        hard_neg_gg_temp.add(tuple(sorted((g2, g3))))
                        hard_neg_gg_temp.add(tuple(sorted((g3, g4))))

                        # gene-pathway & pathway-gene
                        if (g2, p2) not in gp_pairs:
                            hard_neg_gp_temp.add((g2, p2))
                            hard_neg_pg_temp.add((p2, g2))
                        if (g3, p2) not in gp_pairs:
                            hard_neg_gp_temp.add((g3, p2))
                            hard_neg_pg_temp.add((p2, g3))
                        if (g4, p2) not in gp_pairs:
                            hard_neg_gp_temp.add((g4, p2))
                            hard_neg_pg_temp.add((p2, g4))

                    for gi in sampled_genes:
                        trialsi = 0
                        samplei = 0
                        while (trialsi < 50 and samplei < 1):
                            trialsi += 1
                            jump_pathways = self._sample_pathways_jump(gi, p_id, k=3)
                            checkin = 0
                            for p2 in jump_pathways:
                                if (gi, p2) not in gp_pairs:
                                    hard_neg_gp_temp.add((gi, p2))
                                    hard_neg_pg_temp.add((p2, gi))
                                    checkin += 1
                            if checkin > 0:
                                samplei = 2
                            
            # Final selection
            if len(hard_neg_gg_temp) > target_hard_neg_gg:
                hard_neg_gg = set(random.sample(list(hard_neg_gg_temp), target_hard_neg_gg))
            else:
                hard_neg_gg = hard_neg_gg_temp

            if len(hard_neg_gp_temp) > target_hard_neg_gp:
                hard_neg_gp = set(random.sample(list(hard_neg_gp_temp), target_hard_neg_gp))
                hard_neg_pg = set((pi, gi) for gi, pi in hard_neg_gp)
            else:
                hard_neg_gp = hard_neg_gp_temp
                hard_neg_pg = set((pi, gi) for gi, pi in hard_neg_gp)


        medium_neg_gg = set()
        medium_neg_gp = set()
        medium_neg_pg = set()

        medium_neg_gg_temp = set()
        medium_neg_gp_temp = set()
        medium_neg_pg_temp = set()
        # Sample medium negatives
        if medium_refs:
            for ref in medium_refs:
                # Identify  pathways
                ref_pathway_items = list(ref.pathway_genes.items())
                # Go over all pathways in this medium negative sample
                for p_id, p_data_ref in random.sample(ref_pathway_items, len(ref_pathway_items)):
                    enriched_genes = set(p_data_ref["gene"])
                    if len(enriched_genes) == 0:
                        continue

                    # Existence check
                    if p_id in ctx.pathway_genes:
                        query_p_data = ctx.pathway_genes[p_id]
                        query_p_genes = set(query_p_data["genes"])

                        # Similarity check
                        intersection = len(query_p_genes.intersection(enriched_genes))
                        union = len(query_p_genes.union(enriched_genes))
                        pathway_sim = intersection / union if union > 0 else 0

                        # If the pathway is effectively the same in both, skip it
                        if pathway_sim > 0.6:
                            continue

                    pathway_genes = self.pathway_ppi_genes.get(p_id, [])
                    non_enriched_genes = [gi for gi in pathway_genes if gi not in enriched_genes]

                    trials = 0
                    max_trials = 1000
                    sampled_genes = set()

                    while (
                        len(sampled_genes) < max_genes_per_pathway
                        and trials < max_trials
                    ):
                        trials += 1

                        # prioritize enriched genes
                        if random.random() < 0.8 or not non_enriched_genes:
                            g0 = random.choice(enriched_genes)
                        else:
                            g0 = random.choice(non_enriched_genes)

                        nbr1 = [gi for gi in self._gene_neighbors(g0, p_id) if gi != g0]
                        
                        if not nbr1:
                            continue

                        g1 = random.choice(nbr1)

                        edge01 = tuple(sorted((g0, g1)))

                        if (edge01 in gg_pairs or edge01 in hard_neg_gg):
                            continue

                        nbr2 = [gi for gi in self._gene_neighbors(g1, p_id) if gi != g0 and gi != g1]

                        if not nbr2:
                            continue

                        g2 = random.choice(nbr2)

                        edge12 = tuple(sorted((g1, g2)))

                        if (edge12 in gg_pairs or edge12 in hard_neg_gg):
                            continue

                        sampled_genes.update([g0, g1, g2])
                        
                        # gene-gene
                        medium_neg_gg_temp.add(tuple(sorted((g0, g1))))
                        medium_neg_gg_temp.add(tuple(sorted((g1, g2))))

                        # gene-pathway & pathway_gene
                        if (g0, p_id) not in gp_pairs and (g0, p_id) not in hard_neg_gp:
                            medium_neg_gp_temp.add((g0, p_id))
                            medium_neg_pg_temp.add((p_id, g0))
                        if (g1, p_id) not in gp_pairs and (g1, p_id) not in hard_neg_gp:
                            medium_neg_gp_temp.add((g1, p_id))
                            medium_neg_pg_temp.add((p_id, g1))
                        if (g2, p_id) not in gp_pairs and (g2, p_id) not in hard_neg_gp:
                            medium_neg_gp_temp.add((g2, p_id))
                            medium_neg_pg_temp.add((p_id, g2))

                        # pathway jump
                        p2 = self._sample_pathway_jump(g2, p_id)

                        if p2 is None:
                            continue

                        nbr3 = [gi for gi in self._gene_neighbors(g2, p2) if gi != g2]

                        if not nbr3:
                            continue

                        g3 = random.choice(nbr3)

                        edge23 = tuple(sorted((g2, g3)))
                        if (edge23 in gg_pairs or edge23 in hard_neg_gg):
                            continue
                        
                        nbr4 = [gi for gi in self._gene_neighbors(g3, p2) if gi != g2 and gi != g3]

                        if not nbr4:
                            continue

                        g4 = random.choice(nbr4)

                        edge34 = tuple(sorted((g3, g4)))
                        if (edge34 in gg_pairs or edge34 in hard_neg_gg):
                            continue

                        medium_neg_gg_temp.add(tuple(sorted((g2, g3))))
                        medium_neg_gg_temp.add(tuple(sorted((g3, g4))))

                        # gene-pathway & pathway-gene
                        if (g2, p2) not in gp_pairs and (g2, p2) not in hard_neg_gp:
                            medium_neg_gp_temp.add((g2, p2))
                            medium_neg_pg_temp.add((p2, g2))
                        if (g3, p2) not in gp_pairs and (g3, p2) not in hard_neg_gp:
                            medium_neg_gp_temp.add((g3, p2))
                            medium_neg_pg_temp.add((p2, g3))
                        if (g4, p2) not in gp_pairs and (g4, p2) not in hard_neg_gp:
                            medium_neg_gp_temp.add((g4, p2))
                            medium_neg_pg_temp.add((p2, g4))

                    for gi in sampled_genes:
                        trialsi = 0
                        samplei = 0
                        while (trialsi < 50 and samplei < 1):
                            trialsi += 1
                            jump_pathways = self._sample_pathways_jump(gi, p_id, k=3)
                            checkin = 0
                            for p2 in jump_pathways:
                                if (gi, p2) not in gp_pairs and (gi, p2) not in hard_neg_gp:
                                    medium_neg_gp_temp.add((gi, p2))
                                    medium_neg_pg_temp.add((p2, gi))
                                    checkin += 1
                            if checkin > 0:
                                samplei = 2
                            
            # Final selection
            if len(medium_neg_gg_temp) > target_medium_neg_gg:
                medium_neg_gg = set(random.sample(list(medium_neg_gg_temp), target_medium_neg_gg))
            else:
                medium_neg_gg = medium_neg_gg_temp

            if len(medium_neg_gp_temp) > target_medium_neg_gp:
                medium_neg_gp = set(random.sample(list(medium_neg_gp_temp), target_medium_neg_gp))
                medium_neg_pg = set((pi, gi) for gi, pi in medium_neg_gp)
            else:
                medium_neg_gp = medium_neg_gp_temp
                medium_neg_pg = set((pi, gi) for gi, pi in medium_neg_gp)


        easy_neg_gg = set()
        easy_neg_gp = set()
        easy_neg_pg = set()

        easy_neg_gg_temp = set()
        easy_neg_gp_temp = set()
        easy_neg_pg_temp = set()
        # Sample easy negatives
        if easy_refs:
            for ref in easy_refs:
                # Identify  pathways
                ref_pathway_items = list(ref.pathway_genes.items())
                # Go over all pathways in this medium negative sample
                for p_id, p_data_ref in random.sample(ref_pathway_items, len(ref_pathway_items)):
                    enriched_genes = set(p_data_ref["gene"])
                    if len(enriched_genes) == 0:
                        continue

                    # Existence check
                    if p_id in ctx.pathway_genes:
                        query_p_data = ctx.pathway_genes[p_id]
                        query_p_genes = set(query_p_data["genes"])

                        # Similarity check
                        intersection = len(query_p_genes.intersection(enriched_genes))
                        union = len(query_p_genes.union(enriched_genes))
                        pathway_sim = intersection / union if union > 0 else 0

                        # If the pathway is effectively the same in both, skip it
                        if pathway_sim > 0.6:
                            continue

                    pathway_genes = self.pathway_ppi_genes.get(p_id, [])
                    non_enriched_genes = [gi for gi in pathway_genes if gi not in enriched_genes]

                    trials = 0
                    max_trials = 1000
                    sampled_genes = set()

                    while (
                        len(sampled_genes) < max_genes_per_pathway
                        and trials < max_trials
                    ):
                        trials += 1

                        # prioritize enriched genes
                        if random.random() < 0.8 or not non_enriched_genes:
                            g0 = random.choice(enriched_genes)
                        else:
                            g0 = random.choice(non_enriched_genes)

                        nbr1 = [gi for gi in self._gene_neighbors(g0, p_id) if gi != g0]
                        
                        if not nbr1:
                            continue

                        g1 = random.choice(nbr1)

                        edge01 = tuple(sorted((g0, g1)))

                        if (edge01 in gg_pairs or edge01 in hard_neg_gg or edge01 in medium_neg_gg):
                            continue

                        nbr2 = [gi for gi in self._gene_neighbors(g1, p_id) if gi != g0 and gi != g1]

                        if not nbr2:
                            continue

                        g2 = random.choice(nbr2)

                        edge12 = tuple(sorted((g1, g2)))

                        if (edge12 in gg_pairs or edge12 in hard_neg_gg or edge12 in medium_neg_gg):
                            continue

                        sampled_genes.update([g0, g1, g2])
                        
                        # gene-gene
                        easy_neg_gg_temp.add(tuple(sorted((g0, g1))))
                        easy_neg_gg_temp.add(tuple(sorted((g1, g2))))

                        # gene-pathway & pathway_gene
                        if (g0, p_id) not in gp_pairs and (g0, p_id) not in hard_neg_gp and (g0, p_id) not in medium_neg_gp:
                            easy_neg_gp_temp.add((g0, p_id))
                            easy_neg_pg_temp.add((p_id, g0))
                        if (g1, p_id) not in gp_pairs and (g1, p_id) not in hard_neg_gp and (g1, p_id) not in medium_neg_gp:
                            easy_neg_gp_temp.add((g1, p_id))
                            easy_neg_pg_temp.add((p_id, g1))
                        if (g2, p_id) not in gp_pairs and (g2, p_id) not in hard_neg_gp and (g2, p_id) not in medium_neg_gp:
                            easy_neg_gp_temp.add((g2, p_id))
                            easy_neg_pg_temp.add((p_id, g2))

                        # pathway jump
                        p2 = self._sample_pathway_jump(g2, p_id)

                        if p2 is None:
                            continue

                        nbr3 = [gi for gi in self._gene_neighbors(g2, p2) if gi != g2]

                        if not nbr3:
                            continue

                        g3 = random.choice(nbr3)

                        edge23 = tuple(sorted((g2, g3)))
                        if (edge23 in gg_pairs or edge23 in hard_neg_gg or edge23 in medium_neg_gg):
                            continue
                        
                        nbr4 = [gi for gi in self._gene_neighbors(g3, p2) if gi != g2 and gi != g3]

                        if not nbr4:
                            continue

                        g4 = random.choice(nbr4)

                        edge34 = tuple(sorted((g3, g4)))
                        if (edge34 in gg_pairs or edge34 in hard_neg_gg or edge34 in medium_neg_gg):
                            continue

                        easy_neg_gg_temp.add(tuple(sorted((g2, g3))))
                        easy_neg_gg_temp.add(tuple(sorted((g3, g4))))

                        # gene-pathway & pathway-gene
                        if (g2, p2) not in gp_pairs and (g2, p2) not in hard_neg_gp and (g2, p2) not in medium_neg_gp:
                            easy_neg_gp_temp.add((g2, p2))
                            easy_neg_pg_temp.add((p2, g2))
                        if (g3, p2) not in gp_pairs and (g3, p2) not in hard_neg_gp and (g3, p2) not in medium_neg_gp:
                            easy_neg_gp_temp.add((g3, p2))
                            easy_neg_pg_temp.add((p2, g3))
                        if (g4, p2) not in gp_pairs and (g4, p2) not in hard_neg_gp and (g4, p2) not in medium_neg_gp:
                            easy_neg_gp_temp.add((g4, p2))
                            easy_neg_pg_temp.add((p2, g4))

                    for gi in sampled_genes:
                        trialsi = 0
                        samplei = 0
                        while (trialsi < 50 and samplei < 1):
                            trialsi += 1
                            jump_pathways = self._sample_pathways_jump(gi, p_id, k=3)
                            checkin = 0
                            for p2 in jump_pathways:
                                if (gi, p2) not in gp_pairs and (gi, p2) not in hard_neg_gp and (gi, p2) not in medium_neg_gp:
                                    easy_neg_gp_temp.add((gi, p2))
                                    easy_neg_pg_temp.add((p2, gi))
                                    checkin += 1
                            if checkin > 0:
                                samplei = 2
                            
            # Final selection
            if len(easy_neg_gg_temp) > target_easy_neg_gg:
                easy_neg_gg = set(random.sample(list(easy_neg_gg_temp), target_easy_neg_gg))
            else:
                easy_neg_gg = easy_neg_gg_temp

            if len(easy_neg_gp_temp) > target_easy_neg_gp:
                easy_neg_gp = set(random.sample(list(easy_neg_gp_temp), target_easy_neg_gp))
                easy_neg_pg = set((pi, gi) for gi, pi in easy_neg_gp)
            else:
                easy_neg_gp = easy_neg_gp_temp
                easy_neg_pg = set((pi, gi) for gi, pi in easy_neg_gp)

        # Topological negative
        # for gene-gene pairs
        pos_gg_count = len(gg_pairs)
        topo_neg_gg = set()

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

                topo_neg_gg.add(edge)
                sampled += 1

        # for gene-pathway pairs
        topo_neg_gp = set()
        topo_neg_pg = set()
        num_pathways = self.g.num_nodes('pathway')
        all_pathways = list(range(num_pathways))

        while len(topo_neg_gp) < target_topo_neg_gp:
            g = random.choice(list(self.gene2direct_pathways.keys()))
            p = random.choice(all_pathways)
            # skip if gene already belongs to pathway or pathway not TAKE
            if p in self.gene2direct_pathways[g]:
                continue
            if not self.g.nodes['pathway'].data['take_mask'][p]:
                continue

            topo_neg_gp.add((g, p))
            topo_neg_pg.add((p, g))

        # Total negatives for the batch
        final_neg_gg = (
            list(hard_neg_gg) + list(medium_neg_gg) + list(easy_neg_gg) + list(topo_neg_gg)
        )

        final_neg_gp = (
            list(hard_neg_gp) + list(medium_neg_gp) + list(easy_neg_gp) + list(topo_neg_gp)
        )

        final_neg_pg = (
            list(hard_neg_pg) + list(medium_neg_pg) + list(easy_neg_pg) + list(topo_neg_pg)
        )

        return {
            "gg_pos": list(gg_pairs),
            "gp_pos": list(gp_pairs),
            "pg_pos": list(pg_pairs),
            "gg_neg": final_neg_gg,
            "gp_neg": final_neg_gp,
            "pg_neg": final_neg_pg,
        }

class AnalyticReplaySampler:
    def __init__(self, pathway_sampler, buffer_size=500, housekeeper_threshold=0.85):
        self.pathway_sampler = pathway_sampler # the workhorse
        self.buffer_size = buffer_size
        self.hk_threshold = housekeeper_threshold

        # Analytic memory
        self.global_gene_counts = defaultdict(int)
        self.total_samples_seen = 0

        # Replay memory
        self.replay_buffer = []

        # Persistent structures
        self.tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))

        # Store the union of genes for each node
        self.branch_signatures = defaultdict(set)

    def _get_node_key(self, level, node_id, path_so_far):
        # Create unique key for caching branch signatures
        prefix = "_".join(map(str, path_so_far))
        return f"L{level}_{prefix}_{node_id}" if prefix else f"L{level}_{node_id}"

    def _refresh_signature(self, level, ctx):
        # Rebuild a branch signature after removal
        # Levels: 0=Organ, 1=Disease, 2=Stimulus, 3=Cell_Type
        path = [ctx.organ_id, ctx.disease_id, ctx.stimulus_id, ctx.cell_type_id]
        node_id = path[level]
        path_so_far = path[:level]
        key = self._get_node_key(level, path[level], path_so_far)

        # Get the specific sub-branch in the tree
        branch = self.tree
        for i in range(level + 1):
            branch = branch.get(path[i], {})

        # Flatten all indices under this specific branch
        all_indices = self._flatten_indices(branch)

        if not all_indices:
            if key in self.branch_signatures:
                del self.branch_signatures[key]
            return

        # 2. Re-calculate the union
        new_sig = set()
        for idx in all_indices:
            ref_ctx = self.replay_buffer[idx]
            new_sig.update(self._get_all_enriched(ref_ctx))

        self.branch_signatures[key] = new_sig

    def _flatten_indices(self, branch):
        # Recursively pull all buffer indices (leaf sets) from a branch
        if isinstance(branch, set):
            return list(branch)

        indices = []
        for sub_branch in branch.values():
            indices.extend(self._flatten_indices(sub_branch))

        return indices


    def _update_tree_node(self, ctx, idx, action="add"):
        # Incrementally update the tree and signatures when a sample is updated from the buffer
        # Levels: 0=Organ, 1=Disease, 2=Stimulus, 3=Cell_Type
        path = [ctx.organ_id, ctx.disease_id, ctx.stimulus_id, ctx.cell_type_id]
        genes = self._get_all_enriched(ctx)

        # 1. Update leaf pointers
        leaf_set = self.tree[path[0]][path[1]][path[2]][path[3]]
        if action == "add":
            leaf_set.add(idx)
        else:
            leaf_set.discard(idx)

        # 2. Update branch signatures
        path_so_far = []
        for level in range(4):
            node_id = path[level]
            key = self._get_node_key(level, node_id, path_so_far)

            if action == "add":
                self.branch_signatures[key].update(genes)
            else:
                self._refresh_signature(level, ctx)
            # Append current node_id to the path for the next level's key    
            path_so_far.append(node_id)

    def _get_all_enriched(self, ctx):
        """
        To extract unique gene IDs from the pathway_genes dict
        """
        all_genes = set()
        for pw_data in ctx.pathway_genes.values():
            all_genes.update(pw_data["genes"])
        return all_genes

    def _get_contrastive_ref(self, current_ctx):
        if not self.replay_buffer:
            return None

        # If the buffer is small, hierarchical search is not meaningful yet
        if len(self.replay_buffer) < 5:
            return random.choice(self.replay_buffer)

        return self._navigate_for_ref(current_ctx)


    def _count_branch(self, branch):
        """
        Recursively counts how many buffer indices are in this branch
        """
        if isinstance(branch, list): return len(branch)
        return sum(self._count_branch(v) for v in branch.values())

    def _pick_random_from_branch(self, branch):
        """
        Recursively flattens indices in a branch and picks one
        """
        def flatten(b):
            if isinstance(b, list): return b
            res = []
            for v in b.values(): res.extend(flatten(v))
            return res
        return random.choice(flatten(branch))


    def _update_buffer(self, ctx):
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(ctx)
            return

        # Build the hierarchy for the current buffer
        # Structure: {organ: {disease: {stimulus: {cell_type: [indices]}}}}
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for i, item in enumerate(self.replay_buffer):
            tree[item.organ_id][item.disease_id][item.stimulus_id][item.cell_type_id].append(i)

        # --- HIERARCHICAL DECISION ENGINE ---
    
        # 1. Check Root (Organ)
        # If the new sample is a brand new organ, replace the most common one
        if ctx.organ_id not in tree:
            target_org = max(tree.keys(), 
                key=lambda k: self._count_branch(tree[k]))
            self.replay_buffer[self._pick_random_from_branch(tree[target_org])] = ctx
            return

        # 2. Check Child (Disease)
        if ctx.disease_id not in tree[ctx.organ_id]:
            # New disease in this organ and replace a redundant disease in the same organ
            target_dis = max(tree[ctx.organ_id].keys(), 
                key=lambda k: self._count_branch(tree[ctx.organ_id][k]))
            self.replay_buffer[self._pick_random_from_branch(tree[ctx.organ_id][target_dis])] = ctx
            return

        # 3. Check Child (Stimulus)
        if ctx.stimulus_id not in tree[ctx.organ_id][ctx.disease_id]:
            # New stimulus for this disease and replace a redundant stimulus
            target_stim = max(tree[ctx.organ_id][ctx.disease_id].keys(),
                key=lambda k: self._count_branch(tree[ctx.organ_id][ctx.disease_id][k]))
            self.replay_buffer[self._pick_random_from_branch(tree[ctx.organ_id][ctx.disease_id][target_stim])] = ctx
            return

        # 4. Check Bottom (Cell Type / scRNA)
        # Priority: scRNA (id > 0) should NEVER be replaced by Bulk (id == 0)
        if ctx.cell_type_id > 0: # This is a rare scRNA sample
            # Find a Bulk sample in the same disease to replace
            bulk_indices = tree[ctx.organ_id][ctx.disease_id][ctx.stimulus_id].get(0, [])
            if bulk_indices:
                self.replay_buffer[random.choice(bulk_indices)] = ctx
                return

        # 5. Final Fallback (Random Refresh)
        # If the state is totally redundant, only replace with a 5% chance
        if random.random() < 0.05:
            idx = random.randint(0, self.buffer_size - 1)
            self.replay_buffer[idx] = ctx

    def _get_branch_signature(self, branch):
        """
        Return the union of all enriched genes in a tree branch
        """
        all_genes = set()
        # If it is a list (leaf), these are the indices in replay buffer
        if isinstance(branch, list):
            for idx in branch:
                ctx_ref = self.replay_buffer[idx]
                all_genes.update(self._get_all_enriched(ctx_ref))
            return all_genes

        # If it is a dict, recurse deeper
        for sub_branch in branch.values():
            all_genes.update(self._get_branch_signature(sub_branch))
        return all_genes

    def _jaccard(self, set_a, set_b):
        if not set_a or not set_b:
            return 0

        intersection = len(set_a.intersection(set_b))
        return intersection / (len(set_a) + len(set_b) - intersection)

    def _navigate_for_ref(self, current_ctx):
        if not self.replay_buffer:
            return None

        current_genes = self._get_all_enriched(current_ctx)

        # Start at the root of persistent tree
        cursor = self.tree
        path_so_far = []

        print(f"\n[DEBUG NAVIGATE] Query: {current_ctx.organ_id} | {current_ctx.disease_id}")
        print(f"Query Genes: {current_genes}")

        # Iterate through the 4 levles
        for level in range(4):
            scores = {}
            level_name = ["Organ", "Disease", "Stimulus", "CellType"][level]

            for node_id, branch in cursor.items():
                if level == 0 and node_id == current_ctx.organ_id:
                    continue

                # Use the hierarchical key to get the cached signature
                sig_key = self._get_node_key(level, node_id, path_so_far)
                sig = self.branch_signatures.get(sig_key, set())

                if sig:
                    scores[node_id] = self._jaccard(current_genes, sig)

            if not scores:
                # Pick a random path
                target_id = random.choice(list(cursor.keys()))
                print(f"  Level {level} ({level_name}): No scores! Randomly picked -> {target_id}")
            else:
                # Sort scores for cleaner debug output
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                print(f"  Level {level} ({level_name}) Candidates:")
                for name, val in sorted_scores:
                    print(f"    - {name}: {val:.4f}")

                # Pick the Hardest path
                target_id = max(scores, key=scores.get)
                print(f"  --> WINNER: {target_id} (Score: {scores[target_id]:.4f})")

            # Move the cursor deeper into the tree
            cursor = cursor[target_id]
            path_so_far.append(target_id)

        # After 4 steps, cursor is the leaf_set
        if not cursor:
            print("  [!] Reach empty leaf, returning random from buffer.")
            return random.choice(self.replay_buffer)

        leaf_idx = random.choice(list(cursor))
        ref_sample = self.replay_buffer[leaf_idx]
        print(f"  FINAL SELECTED REF: Index {leaf_idx} ({ref_sample.organ_id}/{ref_sample.disease_id})")
        return ref_sample

    def _get_indices_from_path(self, path):
        # Helper to walk the tree and return the set of indices at a leaf
        cursor = self.tree
        try:
            for part in path:
                cursor = cursor[part]
            return list(cursor) if isinstance(cursor, set) else []
        except KeyError:
            return []


    def _navigate_for_ref_global(self, current_ctx, n=2, temperature = 0.2):
        # Global softmax over all leaves
        if not self.replay_buffer:
            return None

        current_genes = self._get_all_enriched(current_ctx)
        candidate_indices = []
        candidate_scores = []
        candidate_meta = []

        # 1. Flatten the tree to find all leaf sets
        for sig_key, sig in self.branch_signatures.items():
            if not sig_key.startswith("L3_"):
                continue

            # Extract the organ ID from the key
            path_parts = sig_key.split("_")
            organ_id_in_key = path_parts[1]

            # Rule: skip any leaves that belong to the query's organ
            if organ_id_in_key == current_ctx.organ_id:
                continue

            if sig:
                # Calculate similarity
                score = self._jaccard(current_genes, sig)

                # Retrieve all buffer indices belonging to this leaf context
                indices = self._get_indices_from_path(path_parts[1:])

                for idx in indices:
                    candidate_indices.append(idx)
                    candidate_scores.append(score)
                    # Create meta string
                    ctx = self.replay_buffer[idx]
                    candidate_meta.append(f"{ctx.organ_id}/{ctx.disease_id}")

        if not candidate_indices:
            return random.choice(self.replay_buffer)

        # 2. Probability (softmax)
        score_array = np.array(candidate_scores)

        # Numerical stability: subtract max score
        shifted_scores = (score_array - np.max(score_array)) / temperature
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        # 3. Probabilistic selection
        #selected_idx = np.random.choice(candidate_indices, p=probabilities)
        #winner_ctx = self.replay_buffer[selected_idx]
        actual_n = min(n, len(candidate_indices))
        selected_indices = np.random.choice(
            candidate_indices,
            size=actual_n,
            replace=False,
            p=probabilities
        )

        # Debug info
        print(f"\n{'='*60}")
        print(f"[GLOBAL SOFTMAX] Query: {current_ctx.organ_id}/{current_ctx.disease_id}")
        print(f"Sampling {actual_n} references...")
        print(f"{'Idx':<4} | {'Context':<20} | {'Jaccard':<10} | {'Prob %':<8}")
        print(f"{'-'*60}")
    
        for i in range(len(candidate_indices)):
            idx = candidate_indices[i]
            score = candidate_scores[i]
            meta = candidate_meta[i]
            prob = probabilities[i] * 100

            marker = " [SELECTED]" if idx in selected_indices else ""
            print(f"{idx:<4} | {meta:<20} | {score:.4f}     | {prob:>6.2f}%{marker}")

        print(f"{'-'*60}")
        print("Final Selection Summary:")
        selected_samples = []
        for s_idx in selected_indices:
            s_ctx = self.replay_buffer[s_idx]
            selected_samples.append(s_ctx)
            print(f"  >> Index {s_idx:<3}: {s_ctx.organ_id}/{s_ctx.disease_id}")
        print(f"{'='*60}\n")

        return [self.replay_buffer[idx] for idx in selected_indices]


    def _get_softmax(self, scores, temperature, mode='hard'):
        eval_scores = scores if mode == 'hard' else (1.0 - scores)
        shifted = (eval_scores - np.max(eval_scores)) / temperature
        exp = np.exp(shifted)
        return exp / np.sum(exp)


    def _navigate_for_ref_all(self, current_ctx, n_hard=2, n_medium=1, n_easy=1, temperature=0.2):
        if not self.replay_buffer:
            return [], [], []

        current_genes = self._get_all_enriched(current_ctx)
    
        # Candidate pools
        diff_organ_candidates = []
        same_organ_candidates = []

        # 1. Categorize candidates from the Tree
        for sig_key, sig in self.branch_signatures.items():
            if not sig_key.startswith("L3_"): 
                continue
        
            path_parts = sig_key.split("_")
            organ_id_in_key = path_parts[1]
            indices = self._get_indices_from_path(path_parts[1:])

            score = self._jaccard(current_genes, sig)
        
            # Scenario A: Different Organ (Hard/Medium/Easy-Contextual)
            if organ_id_in_key != current_ctx.organ_id:
                for idx in indices:
                    diff_organ_candidates.append({'idx': idx, 'score': score})
        
            # Scenario B: Same Organ, Different Context (Easy-Organ)
            else:
                for idx in indices:
                    # Ensure it's not the exact same sample
                    if self.replay_buffer[idx].disease_id != current_ctx.disease_id:
                        same_organ_candidates.append({'idx': idx, 'score': score})

        if not diff_organ_candidates:
            return [], [], []


        # 1. Select hard negative: different organ, high similarity score
        diff_scores = np.array([c['score'] for c in diff_organ_candidates])
        diff_indices = [c['idx'] for c in diff_organ_candidates]

        p_hard = self._get_softmax(diff_scores, temperature, mode='hard')
        hard_idx = np.random.choice(diff_indices, size=min(n_hard, len(diff_indices)), replace=False, p=p_hard)
        for i, c_idx in enumerate(diff_indices):
            ctx = self.replay_buffer[c_idx]
            p_val = p_hard[i] * 100
            sel_mark = " [SELECTED]" if c_idx in hard_idx else ""
            tier_label = "HARD"
            print(f"{tier_label:<10} | {c_idx:<4} | {ctx.organ_id:<8} | {ctx.disease_id:<15} | {diff_scores[i]:.4f} | {p_val:>6.2f}%{sel_mark}")
        print(f"---------")

        # 2. Select medium negative: different organ, low similarity score
        p_medium = self._get_softmax(diff_scores, temperature, mode='easy')
        med_idx = np.random.choice(diff_indices, size=min(n_medium, len(diff_indices)), replace=False, p=p_medium)
        for i, c_idx in enumerate(diff_indices):
            ctx = self.replay_buffer[c_idx]
            p_val = p_medium[i] * 100
            sel_mark = " [SELECTED]" if c_idx in med_idx else ""
            tier_label = "MEDIUM"
            print(f"{tier_label:<10} | {c_idx:<4} | {ctx.organ_id:<8} | {ctx.disease_id:<15} | {diff_scores[i]:.4f} | {p_val:>6.2f}%{sel_mark}")
        print(f"---------")

        # 3. Select easy negative: same organ, low similarity score
        easy_idx = []
        if same_organ_candidates:
            same_scores = np.array([c['score'] for c in same_organ_candidates])
            same_indices = [c['idx'] for c in same_organ_candidates]

            p_easy = self._get_softmax(same_scores, temperature, mode='easy')
            easy_idx = np.random.choice(same_indices, size=min(n_easy, len(same_indices)), replace=False, p=p_easy)
            for i, c_idx in enumerate(same_indices):
                ctx = self.replay_buffer[c_idx]
                p_val = p_easy[i] * 100
                sel_mark = " [SELECTED]" if c_idx in easy_idx else ""
                tier_label = "EASY"
                print(f"{tier_label:<10} | {c_idx:<4} | {ctx.organ_id:<8} | {ctx.disease_id:<15} | {same_scores[i]:.4f} | {p_val:>6.2f}%{sel_mark}")
            

        return (
            [self.replay_buffer[i] for i in hard_idx],
            [self.replay_buffer[i] for i in med_idx],
            [self.replay_buffer[i] for i in easy_idx]
        )


    def add_sample(self, ctx):
        if len(self.replay_buffer) < self.buffer_size:
            idx = len(self.replay_buffer)
            self.replay_buffer.append(ctx)
            self._update_tree_node(ctx, idx, "add")
            return f"Added to index {idx}"
        else:
            # Simple replacement for test: replace index 0 to trigger refresh
            target_idx = 0
            old_ctx = self.replay_buffer[target_idx]

            self.replay_buffer[target_idx] = ctx
            
            self._update_tree_node(old_ctx, target_idx, "remove")
            self._update_tree_node(ctx, target_idx, "add")
            return f"Replaced index {target_idx}"



    def observe_and_sample(self, ctx, temperature=0.2):
        """
        Main entry point for training
        """
        # Get all enriched genes
        self.total_samples_seen += 1
        current_enriched = self._get_all_enriched(ctx)

        # 1. Update analytic counts
        for gid in current_enriched:
            self.global_gene_counts[gid] += 1

        # 2. Determine housekeepers (dynamic analytic logic)
        # Identify genes that are too common if we have enough data
        housekeepers = set()
        if self.total_samples_seen > 20: # wait for a small burn-in period
            limit = self.total_samples_seen * self.hk_threshold
            housekeepers = {gid for gid, count in self.global_gene_counts.items()
                    if count > limit}

        # 3. Get hard negatives from replay buffer (contrastive logic)
        num_hard_contexts = 2
        num_medium_contexts = 2
        num_easy_contexts = 1

        hard_refs, med_refs, easy_refs = [], [], [] # Default to empty
        if self.replay_buffer and len(self.replay_buffer) >= num_hard_contexts + num_medium_contexts + num_easy_contexts:
            # Get multiple references using global softmax logic
            hard_refs, med_refs, easy_refs = self._navigate_for_ref_all(
                ctx,
                n_hard=num_hard_contexts,
                n_medium=num_medium_contexts,
                n_easy=num_easy_contexts,
                temperature=temperature
            )
        
        # 4. Call existing workhorse sampler
        # Pass the memory info into original logic
        samples = self.pathway_sampler.sample_pos_neg_context(
            ctx,
            hard_refs=hard_refs,
            medium_refs=medium_refs,
            easy_refs=easy_refs
        )

        # 5. Update reservoir buffer
        self._update_buffer(ctx)

        return samples





