import utils
from utils import GEOContext
from model.sample_test import AnalyticReplaySampler, PathwayNegativeSampler

def create_sample(geo_id, organ, disease, pathways_dict):
    # pathways_dict format: {"PathwayName": {"parent": "...", "genes": {"G1", "G2"}}}
    ctx = GEOContext(geo_id=geo_id, pathway_genes=pathways_dict)

    # Manually map the IDs the Sampler expects
    ctx.organ_id = organ
    ctx.disease_id = disease
    ctx.stimulus_id = "None"
    ctx.cell_type_id = "Bulk"

    return ctx

s1 = create_sample("G1", "Lung", "Cancer", {
    "Cell Cycle": {"parent": "CP", "genes": {"CDK2", "CDK4", "CCNE1", "CCND1", "E2F1"}}
})

s2 = create_sample("G2", "Liver", "Cancer", {
    "Cell Cycle": {"parent": "CP", "genes": {"CDK4", "CDK6", "CCND1"}}
})

s3 = create_sample("G3", "Liver", "Healthy", {
    "PPAR": {"parent": "Met", "genes": {"PPARA", "RXRA", "FABP1", "CPT1A"}}
})

s4 = create_sample("G4", "Brain", "Alz", {
    "Axon": {"parent": "OS", "genes": {"SLIT2", "ROBO1", "EPHA4", "UNC5B"}}
})

s5 = create_sample("G5", "Lung", "Healthy", {
    "Apoptosis": {"parent": "CD", "genes": {"CASP3", "CASP8", "BAX", "BCL2"}}
})

s6 = create_sample("G6", "Kidney", "Healthy", {
    "Filtration": {"parent": "Phys", "genes": {"NPHS1", "NPHS2", "CUBN", "LRP2"}}
})

query_kidney = create_sample("Q1", "Kidney", "Cancer", {
    "Cell Cycle": {"parent": "CP", "genes": {"CDK4", "CCND1", "E2F1", "MKI67", "TOP2A"}}
})

# Path to the required files
ppi_file = 'data/merged_signaling_network_unique.tsv'
pathway_relations_file = 'data/ReactomePathwaysRelation.MMU.gene.txt'
pathway_gene_file = 'data/reactome_pathway_gene.csv'

g, gene2id, pathway2id = utils.load_reactome_mux_graph(
        ppi_file,
        pathway_gene_file,
        pathway_relations_file)

pathwaysampler = PathwayNegativeSampler(g)

sampler = AnalyticReplaySampler(pathwaysampler, buffer_size=10)

print("--- Testing Real GEOContext Integration ---")
sampler.add_sample(s1)
sampler.add_sample(s2)
sampler.add_sample(s3)
sampler.add_sample(s4)
sampler.add_sample(s5)
sampler.add_sample(s6)

print("--- Running Level-by-Level Navigation ---")
n_refs = 2
hard_refs, med_refs, easy_refs = sampler._navigate_for_ref_all(
    query_kidney, n_hard=2, n_medium=2, n_easy=1, temperature=0.2
)

print(f"Query: {query_kidney.organ_id}/{query_kidney.disease_id}")

