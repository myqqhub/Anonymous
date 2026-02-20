# Anonymous
# Bridge-MedDevKG: A Hybrid Knowledge Graph Framework for Medical Device-Patent Linking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20|%20XGBoost-orange)]()

**Bridge-MedDevKG** is a coarse-to-fine framework for cross-domain entity linking between FDA-approved medical devices and USPTO patents. It constructs a high-fidelity Knowledge Graph by fusing domain-adaptive ontology, multi-signal candidate generation, and learned reranking to bridge the severe semantic gap between regulatory and technical documents.

This repository contains the **code**, **released datasets**, and **pre-computed results** for the paper:
> **From Regulatory Approvals to Patents: Cross-Domain Linking for Cardiovascular Device Traceability**

**Datasets disclosed in `data/`:** (1) **Gold standard** — 585 expert-verified device–patent pairs (`gold_standard.parquet`). (2) **Evaluation / baseline data** — 434 FDA PMA documents (`baseline_fda_docs.parquet`), 50K patent subset (`baseline_patents.parquet`), gold links for retrieval baselines (`baseline_gold_links.parquet`), 2,672 samples for cross-encoder comparison (`evaluation_dataset.csv`), gold relation IDs (`gold_rel_ids.csv`), and a 500-row sample of Stage 2 candidates (`sample_links_to_process.parquet`). (3) **Reranker training data** — `training_data_5a.parquet`, the labeled dataset used to train the XGBoost reranker in Step 5a. The KG in this work is the set of device–patent links (e.g. V4_WEIGHTED_LINK); the data in `data/` are exported from that graph (via `code/local_export_all.py`, which reads from an existing Neo4j instance). One large file — `links_to_process.parquet` (~4.4GB) — is generated from the full Neo4j KG and omitted due to size constraints. A representative sample (`sample_links_to_process.parquet`, 500 rows) is provided in `data/` for reference; with the full file, Steps 5a–5c in `hpc_run_all.py` train the reranker and produce the refined link set (the validated KG). With the disclosed datasets, Steps 6, 7, 8, 13, 14 reproduce all paper tables.

---

## Data at a Glance

| Component | Count |
|-----------|-------|
| FDA Cardiovascular PMA Documents | 434 |
| USPTO Patents | 698,191 |
| Companies (normalized) | 29,758 |
| **Gold-Standard Verified Pairs** | **585** |
| Devices with Disclosures | 88 (20.3%) |


---

## 🚀 Key Features

* **MedDevOnto:** Domain-adaptive ontology injecting expert-guided weights into UMLS. Anchor terms (stent, catheter, valve, etc.) receive weight 1.0; generic terms receive 0.1–0.3. Improves recall by **+11.3%** over uniform UMLS weighting.
* **Multi-Signal Candidate Generation:** Fuses company affiliation ($S_{company}$), SBERT vector similarity ($S_{vector}$), and ontology-weighted entity overlap ($S_{entity}$), achieving **98.97% gold recall** at candidate generation (579/585 pairs).
* **Learned Noise Reduction:** BGE-M3 cross-encoder + XGBoost classification (9-dimensional features, 5-fold CV F1=0.931), achieving **50.9% incremental noise reduction**.
* **Gold standard:** 585 expert-verified device–patent pairs from litigation filings, SEC filings, and virtual patent marking — the first such evaluation set for this task.

---

## 🛠️ Architecture

```
Stage 1: MedDevOnto (Domain-Adaptive Ontology)
  ├── Entity extraction: DeepSeek-V3 with schema-constrained prompting
  ├── Quote-then-Verify grounding (98.8% fidelity; 1.2% hallucination rejection)
  └── UMLS mapping: exact match + head-noun fallback (82.9% coverage)

Stage 2: Multi-Signal Candidate Generation
  ├── S_company ∈ {0,20} — 29,758-entity company dictionary (M&A-aware)
  ├── S_vector ∈ [0,65]  — all-mpnet-base-v2 cosine similarity
  └── S_entity           — ontology-weighted overlap (Tier S: exact / Tier A: CUI / Tier B: parent)
      Admission: S(d,p) ≥ 70 OR rescue (Tier-S anchor ≥ 60 OR sim ≥ 0.88)

Stage 3: Learned Noise Reduction (Reranking)
  ├── Cross-Encoder: BGE-M3, 1024-token context
  ├── XGBoost: 9-dim feature vector
  └── Immunity Rules: sim ≥ 0.92 + company match → bypass XGBoost
```

---

## 📁 Repository Structure

```
├── code/
│   ├── prompts/
│   │   ├── prompt_fda_flat.txt          # FDA entity extraction prompt (DeepSeek-V3)
│   │   ├── prompt_patent_flat.txt       # Patent entity extraction prompt (DeepSeek-V3)
│   │   └── prompt_linking.txt           # LLM direct classification prompt (for LLM baseline evaluation)
│   ├── hpc_run_all.py                   # Main pipeline: Steps 5a–14
│   ├── local_export_all.py              # Export Neo4j KG → parquet/csv for HPC
│   │                                    # NOTE: Requires local Neo4j instance
│   └── local_import_results.py         # Import HPC results → Neo4j
│                                        # NOTE: Requires local Neo4j instance
├── data/
│   ├── gold_standard.parquet           # 585 expert-verified device-patent pairs
│   ├── gold_rel_ids.csv                # Gold relation IDs for Stage 3 monitoring
│   ├── evaluation_dataset.csv          # 2,672 samples for cross-encoder eval (Step 6)
│   ├── baseline_fda_docs.parquet       # 434 FDA PMA documents (Step 7)
│   ├── baseline_patents.parquet        # 50,000 patent subset (Step 7)
│   ├── baseline_gold_links.parquet     # Gold links for baseline R@K evaluation
│   ├── training_data_5a.parquet        # Labeled dataset for XGBoost reranker 
│   └── sample_links_to_process.parquet # 500-row sample of Stage 2 candidates
├── results/
│   ├── baseline_results.csv            # Table 2 Group A: retrieval baseline results
│   ├── reranker_comparison.csv         # Table 5 / Table 17: cross-encoder comparison
│   ├── table1_main_results.csv         # Table 3: Stage-2 ablation (R@Gold by signal)
│   └── table2_ablation.csv             # Table 4: fixed thresholds vs. learned fusion
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Reproduction

### Install dependencies
```bash
pip install -r requirements.txt
```

### Reproduce paper tables directly (CPU only, no GPU needed)

```bash
# Table 2 Group A: Retrieval baselines
python code/hpc_run_all.py 7

# Table 3: Stage-2 signal ablation on gold standard
python code/hpc_run_all.py 8

# Table 4: Threshold sensitivity analysis (Appendix G)
python code/hpc_run_all.py 14

# Unified evaluation table (all groups)
python code/hpc_run_all.py 13
```

### Reproduce cross-encoder comparison (Table 5, GPU recommended)
```bash
python code/hpc_run_all.py 6
```

### Run full Stage 3 pipeline (A800/A100 GPU required)
```bash
# Requires training_data_5a.parquet and links_to_process.parquet
# (generated by local_export_all.py from a Neo4j KG instance)
python code/hpc_run_all.py 5a 5b 5c
```

>  **Note:** `links_to_process.parquet` (~4.4GB) is generated from the full Neo4j KG and 
> omitted due to size constraints. A representative sample (`sample_links_to_process.parquet`, 
> 500 rows) is provided in `data/` for reference. All paper tables can be fully reproduced 
> using the provided `data/` files by running Steps 6, 7, 8, 13, 14.

---

## 📂 Data Sources

### FDA PMA Documents
- **Source:** [FDA PMA Database](https://www.fda.gov/medical-devices/device-approvals-denials-and-clearances/pma-approvals)
- **Scope:** 434 Class III cardiovascular PMA approvals (1976–2024)
- **Filtering:** Product codes (DXY, LWS, NKE, PAQ, NPT, NIQ, MIH, LJP, MIP, MAJ, LWP, MHD, DZN, NKM, DQY, DRB, LOF, DRF, DQE, DTG, DTR, DSY, etc.) + cardiovascular keyword matching + manual exclusion of 29 non-cardiovascular entries (see Appendix A)

### USPTO Patents
- **Source:** [USPTO PatentsView](https://patentsview.org/) (1976–October 2024)
- **Scope:** 698,191 utility patents from 11.2M records (6.2% retention)
- **Filtering:** CPC classifications (A61F2, A61M25, A61B5/6/8, A61B17/34, A61L31/27, etc.) + keyword confirmation for broad manufacturing classes (B23P, C25D, etc.)

### Gold Standard Construction (Appendix I)
- **Sources:** Patent litigation filings (PACER, ITC Section 337), virtual patent marking pages, SEC 10-K/8-K filings, investor presentations, FDA PMA prior art citations
- **Coverage:** 88 devices (20.3% of 434), 585 verified pairs (median 5 / max 83 patents per device)
- **Quality:** All pairs derived from legally binding corporate disclosures — no manual labeling required

### Company Normalization Dictionary
- **Coverage:** 29,758 entities (subsidiaries, historical names, abbreviations)
- **Key M&A examples:** Abbott ← St. Jude Medical (2016, $25B), Abbott ← CardioMEMS (2014), Medtronic ← Covidien (2015, $50B), Boston Scientific ← Guidant (2006, $27B)

---

## 📊 Schema: gold_standard.parquet

| Field | Type | Description |
|-------|------|-------------|
| `fda_id` | str | FDA PMA identifier (e.g., `P030031`) |
| `pat_id` | str | USPTO patent identifier |
| `is_valid` | bool | Has embeddings, entities, and company info |
| `kg_linked` | bool | Linked by Stage 2 rule-based pipeline |
| `comp_d` / `comp_p` | str | Normalized company names |
| `vec_d` / `vec_p` | str | all-mpnet-base-v2 embeddings (comma-separated float) |
| `ent_names_d` / `ent_names_p` | str | Extracted entities (\|\|\|-delimited) |


---

## ⚙️ Configuration

Key thresholds in `hpc_run_all.py` (calibrated on gold-pair similarity distribution; see Appendix G)

---

## 🔧 Entity Extraction Prompts (code/prompts/)

| File | Persona |
|------|---------|
| `prompt_fda_flat.txt` | "Precision Extractor" |
| `prompt_patent_flat.txt` | "Patent Decoder" |
| `prompt_linking.txt` | Judge |

All prompts enforce: (1) in-context normalization rules, (2) Anti-Super-Node constraint (generic terms like "System", "Device" forbidden unless modified), (3) Quote-then-Verify grounding.

---

## 📋 Pre-computed Results (results/)

| File | Paper Table | Description |
|------|------------|-------------|
| `baseline_results.csv` | Table 2, Group A | R@10/100/500, MRR for TF-IDF, BM25, SBERT, BioBERT, SapBERT, Company on 50K patent subset |
| `reranker_comparison.csv` | Table 5 / Table 17 | ROC-AUC, PR-AUC, F1, latency for 13 cross-encoder models on 2,672 evaluation samples |
| `table1_main_results.csv` | Table 3 | R@Gold with 95% CI for Stage-2 signal ablation (w/o Company/Vector/Entity/Rescue) |
| `table2_ablation.csv` | Table 4 | R@Gold and noise reduction for fixed thresholds (θ=60/70/80/90) vs. Stage 3 learned fusion |
