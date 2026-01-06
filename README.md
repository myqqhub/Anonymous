# Anonymous
# Bridge-MedDevKG: A Hybrid Knowledge Graph Framework for Medical Device-Patent Linking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20|%20XGBoost-orange)]()

**Bridge-MedDevKG** is a coarse-to-fine framework designed to bridge the semantic gap between regulatory documents (FDA PMA) and intellectual property (USPTO patents). It constructs a high-fidelity Knowledge Graph linking medical devices to their underlying patents, addressing challenges like vocabulary mismatch ("semantic gap") and many-to-many entity alignment.

This repository contains the code and benchmark data for the paper:  
> **Bridging Regulatory and Intellectual Property Domains: A Hybrid Knowledge Graph Construction Framework for Medical Device-Patent Linking** > 

---

## 🚀 Key Features

* **MedDevOnto:** A domain-adaptive ontology that injects expert-guided weights into UMLS to prioritize device-critical terms (e.g., "stent" vs. "device").
* **Multi-Signal Candidate Generation:** A retrieval module fusing company affiliation, vector similarity (SBERT), and ontology-weighted entity overlap to achieve high recall.
* **Learned Noise Reduction:** A heterogeneous reranking stage combining Cross-Encoders (BGE-M3) and XGBoost classification to filter noise with high precision.
* **New Benchmark:** Includes a gold standard of **584 expert-verified device-patent pairs** for cardiovascular devices, curated from corporate disclosures and litigation filings.

---

## 🛠️ Architecture

The pipeline consists of three main stages:

1.  **Stage 1: Domain-Adaptive Ontology (MedDevOnto)** Extracts entities using schema-guided prompting (DeepSeek-V3) and maps them to UMLS with domain-specific weighting.
2.  **Stage 2: Multi-Signal Fusion** Generates a broad candidate pool by combining:
    * $S_{company}$: Corporate structure matching (handling M&A).
    * $S_{vector}$: Dense retrieval using `all-mpnet-base-v2`.
    * $S_{entity}$: Ontology-weighted overlap.
3.  **Stage 3: Learned Reranking** Refines candidates using a Cross-Encoder (`BGE-reranker-v2-m3`) and a gradient-boosted classifier (XGBoost) trained on hard negatives.
