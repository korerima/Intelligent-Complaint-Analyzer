# 🧠 Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) chatbot to help CrediTrust Financial extract insights from millions of customer complaints.

---

## 💼 Business Context

CrediTrust is a fast-growing digital finance company operating across East Africa. The company receives millions of consumer complaints across five major financial products. Internal teams currently struggle to make sense of this unstructured data.

This project aims to build an internal tool to:
- Help product managers like Asha detect issues in minutes (not days)
- Empower non-technical users to ask natural-language questions
- Turn complaint data into a strategic feedback loop

---

## 📌 Project Objectives

- Load, clean, and filter CFPB complaint data
- Convert narratives into semantically searchable chunks
- Embed text and index it in a vector store (FAISS)
- Lay the foundation for a question-answering RAG pipeline

---

## ✅ Completed Tasks

### Task 1: EDA & Preprocessing
- Loaded CFPB dataset (sample used for dev, full dataset for final run)
- Analyzed product distribution and narrative length
- Filtered records to 5 key product categories:
  - Credit card
  - Personal loan
  - Buy Now, Pay Later (BNPL)
  - Savings account
  - Money transfer
- Cleaned the complaint narratives (lowercase, special character removal)
- ✅ Output: `data/filtered_complaints.csv`

### Task 2: Chunking, Embedding & Indexing
- Split long complaint texts using `LangChain`'s `RecursiveCharacterTextSplitter`
- Used `sentence-transformers/all-MiniLM-L6-v2` for sentence embeddings
- Indexed embeddings with FAISS (`IndexFlatL2`)
- Stored metadata (product type, complaint ID) with each chunk
- ✅ Output: `vector_store/complaint_index.faiss` and `metadata.pkl`

---

## 🔧 Tech Stack

| Component      | Tool / Library                         |
|----------------|----------------------------------------|
| Language Model | sentence-transformers (MiniLM-L6-v2)   |
| Chunking       | LangChain                              |
| Vector Store   | FAISS                                  |
| Visualization  | Matplotlib, Seaborn                    |
| Platform       | Google Colab                           |

---

## 📂 Folder Structure

