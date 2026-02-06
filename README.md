# 🚨 Bitcoin Anti-Money Laundering. Graph Neural Network Classifier

Deep Learning Model • FastAPI • Docker • PyTorch Geometric

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

### 📌 Overview

This project implements an end-to-end **Bitcoin anti-money laundering (AML)** system powered by Graph Neural Networks (GNNs).
It analyzes the **relationship structure** between transactions rather than treating them in isolation, using a **Chebyshev Graph Convolutional Network (ChebNet)** to classify nodes as **Licit** or **Illicit**.
The entire solution is deployed as a Dockerized **FastAPI** application, ready for local or cloud deployment.

### 🔍 Problem Statement
Money laundering patterns on Bitcoin are becoming increasingly sophisticated.
Traditional tabular classifiers often miss "guilt-by-association" patterns where illicit funds move through complex sub-graphs.

This project provides a graph-based fraud detection pipeline that:
* Leverages transaction topology to detect hidden laundering rings.
* Assigns real-time probability scores based on neighbor behavior.
* Enables automated flagging of suspicious transaction chains.
* Supports compliance teams and crypto forensic analysts.

---

### Inspiration & Methodology

This project implements the architecture proposed in **"Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification"** (Ferretti, D’Angelo, & Ghini, 2025).

**How I Digested the Paper:**
While reading the research, I realized that traditional AML methods treat transactions as independent rows in a spreadsheet (Tabular Data). However, money laundering is inherently social—it involves flows between actors.

To replicate and improve upon this, I shifted the paradigm from **Feature Engineering** to **Graph Topology Learning**:
1.  **The Shift:** Instead of asking "Does this transaction look weird?", I built a system that asks "Does this transaction hang out with bad crowds?"
2.  **The Architecture:** I selected **ChebNet (Chebyshev Spectral Graph Convolutions)** because it effectively captures local neighborhood structures (k-hops) without the massive computational overhead of global spectral methods.
3.  **The Result:** The model learns that a "clean" looking transaction is actually "illicit" if it receives funds from a mixer 2 hops away—something a Random Forest model might miss.

---

### 🧩 Dataset & Feature Description

The model is trained on the **Elliptic Data Set**, representing a graph of Bitcoin transactions. Unlike standard datasets, this uses **165 anonymized features** capturing both local details and neighborhood context.

| Feature Type | Feature Name | Description |
| :--- | :--- | :--- |
| **Graph Topology** | `edge_index` | Defines the flow of Bitcoin from one transaction to another (Inputs/Outputs). |
| **Numerical** | `local_feature_1` to `94` | Direct properties of the transaction (time step, fee, inputs/outputs, etc.). |
| **Numerical** | `aggregate_feature_1` to `72` | Aggregated statistics from one-hop backward/forward neighbors (neighbor mean, max, etc.). |
| **Target** | `class` | **1** = Illicit (Money Laundering), **2** = Licit (Safe). |

These features feed into the GNN to learn spatial dependencies between nodes.

### 🧠 Model Summary

* **Model:** 2-Layer Chebyshev Graph Convolutional Network (ChebNet)
* **Output:** `illicit_probability` (0–1), `licit_probability` (0–1)
* **Prediction:** `LICIT` or `ILLICIT`
* **Confidence:** High-precision probability score.

---


### ⚖️ The "Consciousness" Mechanism (Class Imbalance)

A critical challenge in the Elliptic dataset is that **illicit transactions are extremely rare** (<2% of data). A standard model could achieve 98% accuracy simply by closing its eyes and guessing "Licit" for everything—but it would catch zero criminals.

To solve this, I implemented a **Weighted Cross-Entropy Loss** mechanism (inspired by the "Artificial Consciousness" concept in Weber et al.):

* **The Logic:** The model is penalized **3x harder** for missing an illicit transaction than for misclassifying a licit one.
* **The Code:** `class_weights = torch.tensor([1.0, 3.0])`
* **The Result:** This forces the GNN to be "hyper-aware" of the minority class, significantly improving Recall (the ability to actually find the thieves) even if it slightly lowers overall precision.


### 🌍 Real-World Application: CEX & AML Compliance

This system is designed to act as a **Risk Engine Middleware** for Centralized Exchanges (CEX) or Compliance Firms. Here is the operational workflow:

#### 1. The Gatekeeper (Deposit Screening)
* **Scenario:** A user deposits 5 BTC into an Exchange (like Binance or Coinbase).
* **Process:** Before crediting the user's balance, the Exchange backend extracts the transaction features and hits this API.
* **Outcome:** If the model returns `Illicit Confidence > 90%`, the deposit is automatically frozen for manual review, preventing the exchange from unknowingly laundering stolen funds.

#### 2. The Investigator (Forensic Analysis)
* **Scenario:** A hack occurs, and funds are moving rapidly through hundreds of wallets.
* **Process:** An AML Analyst runs this model on the graph of the stolen funds.
* **Outcome:** The model identifies the "Cash-Out" points (exchanges or mixers) by tracing the illicit pattern through the graph, allowing law enforcement to subpoena the correct entities.

#### 3. Automated SAR Filing
* **Scenario:** Regulatory compliance requires reporting suspicious activity.
* **Process:** The system flags high-probability transactions daily.
* **Outcome:** Automatically generates data for **Suspicious Activity Reports (SARs)**, reducing the manual workload for compliance officers.

---

### 💻 Tech Stack

* **Modeling:** PyTorch, PyTorch Geometric, pandas, scikit-learn
* **Backend:** FastAPI
* **Serialization:** PyTorch State Dict (`.pth`)
* **Environment:** Virtualenv (venv)
* **Containerization:** Docker

### Conclusion

This Bitcoin AML platform provides a fully operational, containerized Graph Machine Learning pipeline. By moving beyond simple feature analysis to **graph topology learning**, it delivers superior detection capabilities suitable for:
* Crypto Exchange (CEX) compliance systems
* Forensic blockchain investigations
* Real-time transaction monitoring
* Regulatory reporting automation

This system represents a shift from static rules to dynamic, graph-aware intelligence in fighting financial crime.

Thanks for exploring!
