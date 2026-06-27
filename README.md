\# AML Transaction Monitoring — Self-Learning RAG Pipeline



A portfolio project demonstrating how Retrieval-Augmented Generation (RAG) 

can reduce LLM token costs in a transaction monitoring pipeline.



\## What This Project Does

\- Classifies AML transactions against known fraud typologies using RAG

\- Calls an LLM only when RAG confidence is below threshold (exception cases)

\- Writes LLM answers back to the vector database — system learns over time

\- Tracks and visualizes simulated token cost savings on a live dashboard



\## Stack

\- \*\*Embeddings:\*\* sentence-transformers (all-MiniLM-L6-v2)

\- \*\*Vector DB:\*\* ChromaDB

\- \*\*Pipeline Storage:\*\* SQLite

\- \*\*LLM Exception Handler:\*\* Groq Free Tier (Llama 3)

\- \*\*Cost Simulation:\*\* tiktoken + Haiku pricing rates

\- \*\*Dashboard:\*\* Streamlit

\- \*\*Dataset:\*\* IBM Synthetic AML (Kaggle)



\## Project Documentation

See `docs/DECISION\_JOURNAL.docx` for the full architectural decision log —

every option considered, rejected, and chosen with reasoning.



\## Status

 In Progress — built step by step with full commit history

