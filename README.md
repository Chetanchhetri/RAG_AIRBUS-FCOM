🚀 Airbus A320 RAG — Domain-Specific Embedding Fine-Tuning

This repository implements a domain-adaptive embedding fine-tuning pipeline to improve document retrieval quality in a Retrieval-Augmented Generation (RAG) system built on Airbus A320 technical documentation.

The project focuses on representation learning, not content generation.
Transformer-based embedding models are fine-tuned using abstracted, non-verbatim QA pairs to achieve high-precision semantic search in a safety-critical aviation domain.

✈️ Project Overview

Retrieval quality is a primary bottleneck in RAG systems.
Generic embedding models often fail to capture domain-specific terminology and structure, particularly in highly technical corpora such as aviation manuals.

This project addresses that gap by:

Adapting embeddings to aviation-specific language

Improving retrieval recall without modifying generative models

Maintaining strict compliance and reproducibility standards

🏗️ System Architecture
End-to-End RAG Pipeline
flowchart TD
    A[Airbus A320 Technical Documents] --> B[Text Cleaning & Chunking]

    B --> C[Abstract QA Pair Generation]
    C --> D[QA Dataset<br/>(Query, Positive, Negative)]

    D --> E[Embedding Fine-Tuning<br/>MiniLM Transformer Encoder]
    E --> F[Fine-Tuned Embedding Model]

    F --> G[FAISS Vector Index]

    H[User Query] --> I[Query Embedding]
    I --> F
    F --> G

    G --> J[Top-K Relevant Chunks]
    J --> K[LLM with Prompt Grounding]
    K --> L[Grounded Answer]


Key design choice:
Fine-tuning is applied only at the embedding layer, keeping inference lightweight and stable while improving downstream grounding.

🔍 What This Project Does

Generates abstract, non-verbatim QA pairs from processed Airbus A320 documentation

Fine-tunes a transformer-based embedding model (MiniLM) on domain-specific data

Indexes embeddings using FAISS for fast similarity search

Evaluates retrieval improvements using Recall@k metrics

Integrates seamlessly into existing RAG pipelines

🧠 Key Features

✅ Safe QA pair generation (no proprietary text reproduction)

✅ Transformer-based embedding fine-tuning

✅ Hard-negative mining for contrastive learning

✅ Quantitative retrieval evaluation (Recall@k)

✅ Production-aligned RAG integration
