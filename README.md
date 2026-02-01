🚀 Airbus A320 RAG – Domain-Specific Embedding Fine-Tuning

This repository contains a safe, domain-adaptive embedding fine-tuning pipeline built on official Airbus A320 technical documentation, designed to improve retrieval quality in a Retrieval-Augmented Generation (RAG) system.

The project focuses on representation learning, not content generation, and demonstrates how transformer-based embedding models can be fine-tuned using abstracted QA pairs for high-precision semantic search in safety-critical domains.

✈️ Project Overview

Modern RAG systems heavily depend on embedding quality for accurate document retrieval.
Generic embedding models often underperform on domain-specific technical language, such as aviation manuals.



flowchart TD
    A[Airbus A320 Technical Documents] --> B[Text Cleaning & Chunking]

    B --> C[Abstract QA Pair Generation]
    C --> D[QA Dataset (Query, Positive, Negative)]

    D --> E[Embedding Fine-Tuning<br/>MiniLM Transformer Encoder]

    E --> F[Fine-Tuned Embedding Model]

    F --> G[FAISS Vector Index]

    H[User Query] --> I[Query Embedding]
    I --> F
    F --> G

    G --> J[Top-K Relevant Chunks]

    J --> K[LLM with Prompt Grounding]
    K --> L[Grounded Answer]





This project:

Generates abstract, non-verbatim QA pairs from processed Airbus A320 documents

Fine-tunes a transformer-based embedding model (MiniLM) on this domain data

Integrates the fine-tuned embeddings into a FAISS-based vector search pipeline

Evaluates improvements using Recall@k metrics

🧠 Key Features

✅ Safe QA pair generation (no verbatim content reproduction)

✅ Transformer-based embedding fine-tuning

✅ Hard-negative mining for improved contrastive learning

✅ Quantitative evaluation of retrieval performance

✅ Plug-and-play compatibility with existing RAG pipelines
