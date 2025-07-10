# ====================================
# 🔧 Install Gradio (if in Colab)
# ====================================
# !pip install gradio

# ====================================
# 🧠 RAG-powered Chat Interface
# ====================================
import gradio as gr

# Reuse your existing functions:
# - retrieve_similar_chunks(question)
# - build_prompt(context_chunks, question)
# - generate_answer_llama(prompt)

def rag_chatbot(user_question):
    # Step 1: Retrieve context
    chunks = retrieve_similar_chunks(user_question, k=5)
    
    # Step 2: Build prompt
    prompt = build_prompt(chunks, user_question)
    
    # Step 3: Generate LLM response
    answer = generate_answer_llama(prompt)
    
    # Step 4: Format sources
    sources = "\n\n".join([f"• {chunk['text'][:300]}..." for chunk in chunks])
    
    return answer, sources

# ====================================
# 🎛️ Gradio UI Setup
# ====================================
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 CrediTrust Complaint Analyzer (RAG Chatbot)")
    gr.Markdown("Ask a question about any financial product to explore customer complaints.")

    with gr.Row():
        user_input = gr.Textbox(placeholder="E.g. Why are users unhappy with BNPL?", label="Ask a question")
    
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        answer_box = gr.Textbox(label="AI-generated Answer")
    
    with gr.Accordion("Retrieved Complaint Contexts", open=False):
        context_box = gr.Textbox(label="Source Complaints", lines=10)

    def handle_question(q):
        answer, sources = rag_chatbot(q)
        return answer, sources

    submit_btn.click(handle_question, inputs=[user_input], outputs=[answer_box, context_box])
    clear_btn.click(lambda: ("", "", ""), outputs=[user_input, answer_box, context_box])

# ====================================
# 🚀 Launch App
# ====================================
demo.launch(debug=True, share=True)