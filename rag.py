from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# ----------------- Load Documents -----------------
def load_documents(file_path):
    df = pd.read_excel(file_path)
    questions = df['Question'].astype(str).tolist()
    answers = df['Answer'].astype(str).tolist()
    return questions, answers

# ----------------- Embedding -----------------
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_documents(docs, model):
    embeddings = model.encode(docs, show_progress_bar=False)
    return np.array(embeddings)

# ----------------- FAISS Index -----------------
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ----------------- Retrieve Answers and Suggestions -----------------
def retrieve_top_k_answers(query, model, index, questions, answers, k=3):
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, k)

    retrieved_answers = []
    retrieved_questions = []
    for idx in I[0]:
        if idx < len(answers):
            ans = answers[idx]
            que = questions[idx]
            if ans not in retrieved_answers:
                retrieved_answers.append(ans)
            if que not in retrieved_questions:
                retrieved_questions.append(que)
    return retrieved_answers, retrieved_questions

# ----------------- Setup -----------------
DOCUMENT_FILE = "E:\\Infinitude IT\\infinitude rag llm chatbot\\Infinitude_IT_Data_QA.xlsx"

questions, answers = load_documents(DOCUMENT_FILE)
embedding_model = get_embedding_model()
question_embeddings = embed_documents(questions, embedding_model)
faiss_index = create_faiss_index(question_embeddings)

# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    user_input = data.get("user_input", "")
    top_answers, _ = retrieve_top_k_answers(user_input, embedding_model, faiss_index, questions, answers, k=3)
    response = "\n\n".join(top_answers)
    return jsonify({"response": response})

@app.route("/get_suggestions", methods=["POST"])
def get_suggestions():
    data = request.get_json()
    partial_input = data.get("user_input", "")
    
    if not partial_input.strip():
        return jsonify({"suggestions": []})
    
    _, top_questions = retrieve_top_k_answers(partial_input, embedding_model, faiss_index, questions, answers, k=5)
    
    return jsonify({"suggestions": top_questions})

if __name__ == "__main__":
    app.run(debug=True)
