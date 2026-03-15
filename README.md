# RAG-Question-Answering-Pipeline
A Python-based Retrieval-Augmented Generation (RAG) project that answers questions over a document corpus using LangChain, OpenAI embeddings, FAISS, and GPT-4o-mini.

Overview
This project loads a document dataset, splits the documents into chunks, stores them in a FAISS vector database, retrieves relevant passages for a user query, and generates a grounded answer using an LLM.
It also includes a simple evaluation module for comparing generated answers against ground-truth answers from benchmark question sets.

Features
- Document ingestion from CSV
- Text chunking with RecursiveCharacterTextSplitter
- Embedding generation with OpenAIEmbeddings
- Vector search with FAISS
- Answer generation with ChatOpenAI
- Source attribution in responses
- Basic evaluation using answer similarity scoring

Dataset Files
https://www.kaggle.com/datasets/samuelmatsuoharris/single-topic-rag-evaluation-dataset
This repo uses four CSV files:
- documents.csv — source documents indexed by the RAG pipeline
- single_passage_answer_questions.csv — questions answerable from one passage
- multi_passage_answer_questions.csv — questions requiring evidence from multiple passages
- no_answer_questions.csv — questions that should return I don't know

How It Works
1. Retrieval Pipeline
rag_pipeline.py:
- loads the documents CSV
- converts rows into LangChain Document objects
- chunks text into smaller passages
- builds a FAISS vector store using OpenAI embeddings
- retrieves relevant chunks for a query

2. Generation
agent.py:
- sends retrieved context and the user question to gpt-4o-mini
- constrains the model to answer only from the provided context
- returns the answer with sources

3. Evaluation
evaluate.py:
- compares generated answers with expected answers
- calculates similarity scores using SequenceMatcher
- returns an average score over the evaluation set
