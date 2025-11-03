# Overview
This project implements a Retrieval-Augmented Generation (RAG) system based on the provided `chat_with_pdf.py` framework and the example langgraph_chroma_retreiver.ipynb.

## Objectives
- Implement a document-based QA system using retrieval and generation.
- Test the system using the sample data file `/data/RAG_source.txt`.
- Adapt the chatbot to follow user prompts and course assignment instructions.

## Key Features
- Retriever: Implemented with `Chroma` vector store and embeddings.
- Data Sources: Supports both `.pdf` and `.txt` inputs.(also support `.md`)
- Chat Interface: Interactive console or API-based chat system.
- RAG Pipeline: Combines document retrieval with LLM-based response generation.
- Prompt-based Evaluation: Allows querying the system using Assignment 1 instructions.

## Implementation Note
- The base structure was adapted from `chat_with_pdf.py`.
- The retrieval logic references `langgraph_chroma_retreiver.ipynb`.
- The `/data/**` files was used to validate retrieval accuracy.(Mainly `RAG_source.txt`.)
- Still under debugging.
