# TASK 3: RAG RETRIEVAL + GENERATION (LLaMA 2)

# Load Vector Store & Metadata
from dotenv import load_dotenv
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load credentials from .hf.env
load_dotenv(dotenv_path="hf.env")

HF_TOKEN = int(os.getenv("TOKEN_API"))

# Load FAISS index and metadata
index = faiss.read_index("vector_store/complaint_index.faiss")
with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieve Top-k Similar Complaint Chunks
def retrieve_similar_chunks(query, k=5):
    query_vec = embed_model.encode([query])
    D, I = index.search(query_vec, k)
    results = [metadata[i] for i in I[0]]
    return results

# Build Prompt Template
def build_prompt(context_chunks, question):
    context = "\n".join([f"- {doc['text']}" for doc in context_chunks])
    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use only the information from the retrieved complaint excerpts below.

Context:
{context}

Question: {question}
Answer:"""
    return prompt.strip()

# Set Up Hugging Face Transformers Pipeline
from huggingface_hub import login
login(new_session=False)
from transformers import pipeline
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

#  Generate Answer
def generate_answer_llama(prompt):
    response = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
    return response[len(prompt):].strip()

# Run Evaluation with 5 Test Questions
test_questions = [
    "Why are people unhappy with Buy Now, Pay Later?",
    "What complaints are common about personal loans?",
    "Do users report fraud issues on savings accounts?",
    "Are credit card users complaining about billing?",
    "Any delays in money transfer services?"
]

for q in test_questions:
    print("Question:", q)
    
    # Retrieval
    chunks = retrieve_similar_chunks(q, k=5)
    print("\nContext Chunks:")
    for i, c in enumerate(chunks):
        print(f"{i+1}. {c['text'][:100]}...\n")
    
    # Prompt and Answer
    prompt = build_prompt(chunks, q)
    answer = generate_answer_llama(prompt)
    
    print("mistralai Answer:\n", answer)
    print("="*100)
