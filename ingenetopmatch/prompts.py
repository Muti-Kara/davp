"""Report-format strings for the Detailed Variant Report (the ``d_i`` document).

These reproduce the exact layout of the ``variant_reports/*.txt`` shipped with the demo,
which is the evidence document inGeneTopMatch assembles for each variant and which Step 2
(``davp.py``) later hands to the LLM via ``FINAL_LLM_PROMPT``. The variant / ClinVar
sub-blocks and their ``====`` underlines match the production
``gagi_service`` ``VARIANT_FORMAT`` / ``CLINVAR_FORMAT`` strings.
"""

# Variant annotation block (x_i). Begins and ends with a blank line, matching the shipped
# reports. Fields mirror the columns present in the demo's variant_reports/*.txt.
VARIANT_FORMAT = """

DETAILED VARIANT INFORMATION:
=============================
• rsID: {rsid}
• Chromosome: {chrom}
• Position: {pos}
• Reference Allele (REF): {ref}
• Alternative Allele (ALT): {alt}
• Genotype: {gt}
• Quality Score: {qual}
• Gene Names: {gene_names}
• Databases: {databases}

CLINICAL ANNOTATIONS:
====================
• ClinVar Significance: {clinvar_sig}
• ACMG Classification: {acmg}
• OMIM: {omim}

POPULATION FREQUENCIES:
======================
• gnomAD Allele Frequency: {gnomad_af}
• gnomADg Allele Frequency: {gnomadg_af}
• 1KG Frequency: {kg_frequency}

SEQUENCING DATA:
================
• Depth (DP): {dp}
• Allelic Depth (AD): {ad}

"""

# ClinVar pathogenicity summary header. Phenotype/disease breakdown is appended after.
CLINVAR_FORMAT = """
CLINVAR PATHOGENICITY SUMMARY:
==============================
• Total ClinVar variants analyzed: {total_variants}
• Pathogenic/Likely Pathogenic: {pathogenic_variants}
• Benign/Likely Benign: {benign_variants}
• Uncertain Significance: {uncertain_variants}
• Other Classifications: {other_variants}
"""

# Wrapper assembling the four sections into the final Detailed Variant Report. The blank-line
# spacing here, combined with the leading/trailing blank lines of VARIANT_FORMAT / CLINVAR_FORMAT
# and the trailing "\n\n" of the GWAS / entity blocks, reproduces the shipped .txt byte layout.
REPORT_HEADER = "=== {variant_id} Genomic Analysis Report ===\n\n\n#VARIANT INFO:\n"
GWAS_HEADER = "\n# GWAS INFO\n"
ENTITY_HEADER = "\n#ENTITY MAPPING:\n"
CLINVAR_HEADER = "\n#CLINVAR INFO:\n"

NO_GWAS_TEXT = "No GWAS associations found to analyze.\n\n"
