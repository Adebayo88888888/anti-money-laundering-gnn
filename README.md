# 🕵️‍♂️ Anti-Money Laundering (AML) with Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

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
