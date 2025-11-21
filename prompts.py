VARIANT_PROMPT = '{rank}. Variant: {chr}:{pos} {ref}>{alt} | Gene: {gene} | GT: {gt} | IDs: {ids} | AF: 1KG={kg}, gnomAD={gad}, gnomADg={gadg} | ACMG: {acmg} | ClinVar: {clinvar} | Scores: SIFT={sift}, Polyphen={polyphen}, CADD={cadd}, REVEL={revel}, SpliceAI={spliceai}, DANN={dann}, MetalR={metalr} | Alpha Missense: {am_score} ({am_pred}) | OMIM: {omim}'

GENE_PROMPT = '{gene_cache} \nVariants on this gene with their ranks: \n{variants}'

PRELIMIN8_PROMPT = """
You are a genetic assistant AI. You will be given genes and the patient information which includes phenotypes of patients as HPO terms, the clinical summary and additional comments. 

IMPORTANT CONTEXT:
==================
The patient has genetic variants in each of the genes listed below. For each gene, you will see:
1. A comprehensive gene report (including disease associations, phenotypes, and clinical significance)
2. A list of the ACTUAL VARIANTS found in this patient within that gene, with detailed information including:
   - Genomic location and allele information
   - Clinical annotations (ClinVar significance, ACMG classification, OMIM status)
   - Population frequencies (gnomAD, 1000 Genomes)

Your task is to rank these eight genes based on their potential significance to the patient's condition.

CRITICAL EVALUATION CRITERIA:
==============================
When ranking each gene, you must consider BOTH:
1. The gene's general disease associations and known phenotypes (from the gene report)
2. The SPECIFIC VARIANTS found in this patient within that gene (variant details provided)

A gene ranks higher if:
- Its known disease phenotypes align well with the patient's symptoms
- The patient's variants in this gene show pathogenic/likely pathogenic ClinVar classifications
- The variants are rare in the population (low gnomAD frequencies)
- The ACMG classification supports pathogenicity

IMPORTANT: Ranking a gene as significant means you believe one or more of the variants in that gene could be contributing to the patient's condition. The gene report provides context, but the actual variants in the patient are what matter for diagnosis.

Patient Information:
{target}

Here are the 8 genes to rank (each with the patient's variants listed):
--- Gene 1 ---
Identifier: {item_1_id}
Report: {item_1_info}
--- Gene 2 ---
Identifier: {item_2_id}
Report: {item_2_info}
--- Gene 3 ---
Identifier: {item_3_id}
Report: {item_3_info}
--- Gene 4 ---
Identifier: {item_4_id}
Report: {item_4_info}
--- Gene 5 ---
Identifier: {item_5_id}
Report: {item_5_info}
--- Gene 6 ---
Identifier: {item_6_id}
Report: {item_6_info}
--- Gene 7 ---
Identifier: {item_7_id}
Report: {item_7_info}
--- Gene 8 ---
Identifier: {item_8_id}
Report: {item_8_info}

Please rank all eight genes from 1 (most significant) to 8 (least significant).
Return JSON with keys 'rank_1'..'rank_8', each containing only 'gene_id' (no explanation).

Return your response as pure JSON only. Do not wrap it in markdown code blocks or use backticks.
"""

FINAL_LLM_PROMPT = """
Background: You are a genomic data analyst AI. Overall goal is to spot pathogenic variants in a patient.
In this step, your job is to create a detailed report about the variant and the genetic entities that are most likely to be affected by the variant.

0a) Previously an LLM has provided us a relevant genetic entities (genes, enhancers, promoters, transcription factors, etc.) that might be affected by a variant in a pathogenic way.
0b) Then we have retrieved other variants on that entity which has known GWAS associations and ClinVar entries.
- Now you will be given the variant and a list of genetic entities, each entity will be provided with a list of GWAS associations and ClinVar data for the variants on that entity.
1) Your job is to summarize the information, about the variant, genetic entities and GWAS associations and ClinVar data of the variants on that entity.
2a) Disease and phenotype information coming from GWAS and ClinVar data will be very important and should be included in the report. Any pattern of phenotypes or diseases about a genetic entity is very important.
2b) Mention all the phenotypes and diseases that are associated.
3) Please mention any clue about the pathogenicity of the variant and its relations to the disease and phenotypes based on GWAS and ClinVar data.
 
The report should include all the important information and should not be too long.

{variant_info_section}
Now create the report:
"""

ELIMIN8_PROMPT = """
You are a genetic assistant AI. You will be given genetic variants (mutations) and the patient information which includes phenotypes of patients as HPO terms, the clinical summary and additional comments. Variants are analyzed and annotated by our systems previous steps and a report was created for them which includes important genetic entities such as genes, transcripts, TF-binding sites, enhancers, ChromHMM states, etc that the variant is located in. Also, other variants' GWAS associations and ClinVar annotations that are on the same entity are included.

Your task is to rank eight genetic variants based on their potential significance to a patient's condition.
When ranking, it is crucial to consider the patient's listed phenotypes, epicrisis and comments. A variant is more significant if its known clinical associations align well with the patient's symptoms. Conversely, a variant is less likely to be the cause if it is strongly associated with phenotypes that the patient does not have.

Patient Information:
{target}

Here are the 8 variants to rank:
--- Variant 1 ---
Identifier: {item_1_id}
Report: {item_1_info}
--- Variant 2 ---
Identifier: {item_2_id}
Report: {item_2_info}
--- Variant 3 ---
Identifier: {item_3_id}
Report: {item_3_info}
--- Variant 4 ---
Identifier: {item_4_id}
Report: {item_4_info}
--- Variant 5 ---
Identifier: {item_5_id}
Report: {item_5_info}
--- Variant 6 ---
Identifier: {item_6_id}
Report: {item_6_info}
--- Variant 7 ---
Identifier: {item_7_id}
Report: {item_7_info}
--- Variant 8 ---
Identifier: {item_8_id}
Report: {item_8_info}

Please rank all eight of the provided variants from 1 (most significant) to 8 (least significant). Each variant must be assigned a unique rank. Do not omit any variants or rank the same variant twice.
Provide your response as a JSON object with exactly eight keys: 'rank_1' through 'rank_8'. Each value must just be the variant's identifier.

Return your response as pure JSON only. Do not wrap it in markdown code blocks or use backticks.
"""

ROUND_ROBIN_PROMPT = """You are a genetic assistant AI. You will be given two genetic variants (mutations) and the patient information which includes phenotypes of patients as HPO terms, the clinical summary and additional comments. Variants are analyzed and annotated by our systems previous steps and a report was created for them which includes important genetic entities such as genes, transcripts, TF-binding sites, enhancers, ChromHMM states, etc that the variant is located in. Also, other variants' GWAS associations and ClinVar annotations that are on the same entity are included.

Additionally, you will be provided with detailed biomedical reports for the genes associated with each variant. These gene reports contain comprehensive information about gene function, disease associations, and clinical significance.

Your task is to compare these two genetic variants and determine which one is more significant for the given patient.

When making your decision, it is crucial to consider the patient's listed phenotypes, epicrisis and comments. A variant is more significant if its known clinical associations align well with the patient's symptoms. Conversely, it is less likely to be the cause if it is strongly associated with phenotypes that the patient does not have.

Patient Information:
{target}

---
{item_1_id}:
Report: {item_1_info}
---
{item_2_id}:
Report: {item_2_info}
---

Please compare these two variants and declare a winner. You must always choose a single winner; ties are not allowed. Provide a score from 0-10 indicating how much more significant the winner is, and a brief explanation for your choice.

Return your response as pure JSON only. Do not wrap it in markdown code blocks or use backticks.
"""