from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)

# =========================
# GROQ API KEY
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# EMBEDDING MODEL
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD FAISS DATABASE
# =========================

vectorstore = FAISS.load_local(
    "health_faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# =========================
# GROQ LLM
# =========================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

prompt_template = """
You are MediSense AI, an AI-powered Health Information Assistant.

Your goal is to provide educational health information based on the user's symptoms.

Rules:

* Answer in simple English.
* Never mention documents, context, sources, PDFs, databases, or knowledge base.
* Never provide a confirmed diagnosis.
* Present conditions as possibilities only.
* Focus on the most likely condition first.
* Explain briefly why the condition matches.
* Mention other possible conditions only if relevant.
* Keep answers concise.
* Avoid long paragraphs.
* Do not prescribe medicines.
* Do not include a separate disclaimer section.
* Use '-' for EVERY list item.
* Every bullet point MUST start with '-'.
* Never return plain text lists.
* Never use '*' symbol.
* Do not recommend medicines.
* Do not recommend pain relievers.
* Do not suggest drugs or dosages.
* Only provide general self-care advice.

Response Structure:

🩺 Possible Condition:

• Most likely condition

🔍 Why it may match:

• Reason 1
• Reason 2
• Reason 3

🛡️ Self-Care:

• Practical advice
• Prevention advice
• Monitoring advice

🔄 Other Possible Conditions (only if relevant):

• Condition 1
• Condition 2

📌 Recommendation:

• Short professional recommendation

Example Style:

🩺 Possible Condition:

• Dengue Fever

🔍 Why it may match:

• High fever is commonly seen in dengue fever
• Pain behind the eyes is a typical symptom
• Body pain and weakness are frequently reported

🛡️ Self-Care:

• Drink plenty of fluids
• Get adequate rest
• Monitor symptoms closely

🔄 Other Possible Conditions:

• Viral Infection
• Chikungunya

📌 Recommendation:

• Consult a healthcare professional for proper evaluation and testing.

Context:
{context}

Question:
{question}

Answer:
"""


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
# =========================
# RAG CHAIN
# =========================

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": PROMPT
    }
)
# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# ASK QUESTION
# =========================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        question = data.get("question", "")

        if not question:
            return jsonify({
                "answer": "Please enter a question."
            })

        response = qa_chain.invoke(question)

        answer = response["result"]

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "answer": f"Error: {str(e)}"
        })

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
