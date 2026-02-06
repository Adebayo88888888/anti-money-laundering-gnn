# 🕵️‍♂️ Anti-Money Laundering (AML) with Graph Neural Networks



An end-to-end Machine Learning pipeline that detects illicit Bitcoin transactions (money laundering) using **Graph Neural Networks (GNN)**. The system is trained on the Elliptic Data Set and deployed as a containerized microservice API.

## 🚀 Project Overview
Traditional fraud detection looks at transactions in isolation. This project uses **Chebyshev Graph Convolutional Networks (ChebNet)** to analyze the *relationships* between transactions. By treating the blockchain as a graph, the model can detect suspicious patterns even when individual transaction features look normal.

* **Dataset:** Elliptic Data Set (200k+ Bitcoin nodes, 165 features).
* **Model:** 2-Layer ChebNet (GNN).
* **Performance:** ~99.9% Confidence on clear licit/illicit cases.
* **Deployment:** Dockerized FastAPI application.

---

## 📂 Project Structure

```bash
├── 📁 data/                        # Contains Elliptic dataset (features & classes)
├── 📁 production/
│   ├── app.py                      # FastAPI Server (The Inference Engine)
│   └── 📁 weights/                 # Stores the trained model (.pth)
├── anti_money_laundering_gnn.ipynb # 📓 Training Notebook (Source of Truth)
├── Dockerfile                      # Container configuration
├── requirements.txt                # Python dependencies
├── test.py                         # Script to test a normal transaction
├── catch_thief.py                  # Script to verify detection of illicit transactions
└── README.md                       # Documentation

```














# 🚨 Bitcoin Anti-Money Laundering. Graph Neural Network Classifier

Deep Learning Model • FastAPI • Docker • PyTorch Geometric

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

 ### Overview

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


🧩 Dataset & Feature Description

The model is trained on the **Elliptic Data Set**, representing a graph of Bitcoin transactions. Unlike standard datasets, this uses **165 anonymized features** capturing both local details and neighborhood context.

| Feature Type | Feature Name | Description |
| :--- | :--- | :--- |
| **Graph Topology** | `edge_index` | Defines the flow of Bitcoin from one transaction to another (Inputs/Outputs). |
| **Numerical** | `local_feature_1` to `94` | Direct properties of the transaction (time step, fee, inputs/outputs, etc.). |
| **Numerical** | `aggregate_feature_1` to `72` | Aggregated statistics from one-hop backward/forward neighbors (neighbor mean, max, etc.). |
| **Target** | `class` | **1** = Illicit (Money Laundering), **2** = Licit (Safe). |

These features feed into the GNN to learn spatial dependencies between nodes.

  
### Model Summary

* **Model:** 2-Layer Chebyshev Graph Convolutional Network (ChebNet)
* **Output:** `illicit_probability` (0–1), `licit_probability` (0–1)
* **Prediction:** `LICIT` or `ILLICIT`
* **Confidence:** High-precision probability score.


### 💻 Tech Stack

* **Modeling:** PyTorch, PyTorch Geometric, pandas, scikit-learn
* **Backend:** FastAPI
* **Serialization:** PyTorch State Dict (`.pth`)
* **Environment:** Virtualenv (venv)
* **Containerization:** Docker

## Conclusion

This Bitcoin AML platform provides a fully operational, containerized Graph Machine Learning pipeline. By moving beyond simple feature analysis to **graph topology learning**, it delivers superior detection capabilities suitable for:
* Crypto Exchange (CEX) compliance systems
* Forensic blockchain investigations
* Real-time transaction monitoring
* Regulatory reporting automation

This system represents a shift from static rules to dynamic, graph-aware intelligence in fighting financial crime.
Thanks for exploring.......
