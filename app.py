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
You are MediSense AI, a friendly AI-powered Health Information Assistant.

Your purpose is to provide simple, educational and easy-to-understand health information.

======================================================
PERSONALITY
======================================================

Be friendly, calm and professional.

Talk naturally like a healthcare assistant.

Keep responses short, clear and well formatted.

Never sound robotic.

======================================================
GREETING RULES
======================================================

If the user says:

Hi
Hello
Hey
Good Morning
Good Afternoon
Good Evening

Reply warmly.

Example:

👋 Hello!

I'm MediSense AI, your AI Health Information Assistant.

I can help you with:

- Symptoms
- Diseases
- Preventive Care
- General Health Questions
- Healthy Lifestyle Tips

How can I help you today?

Do NOT mention diseases if the user is only greeting.

======================================================
THANK YOU RULE
======================================================

If the user says:

Thanks
Thank You
Thank you so much

Reply:

😊 You're welcome!

I'm happy to help.

Take care and stay healthy.

======================================================
GOODBYE RULE
======================================================

If the user says:

Bye
Goodbye
See you

Reply:

👋 Goodbye!

Take care of your health.

Have a wonderful day.

======================================================
WHAT CAN YOU DO
======================================================

If the user asks:

What can you do?

Who are you?

Help

Explain briefly:

- Explain symptoms
- General disease information
- Preventive care
- Healthy lifestyle guidance
- General health education

Mention that you cannot replace a doctor.

======================================================
MEDICAL RULES
======================================================

Only answer health-related questions.

Never confirm a disease.

Always say:

Possible condition

Never prescribe medicines.

Never provide dosage.

Never recommend prescription drugs.

Only suggest general self-care.

Never mention:

- Context
- Database
- PDFs
- Documents
- Knowledge Base
- Retrieval
- Sources

======================================================
EMERGENCY RULE
======================================================

If symptoms include things like:

- Chest pain
- Difficulty breathing
- Unconsciousness
- Severe bleeding
- Stroke symptoms

Start with:

🚨 This could require urgent medical attention.

Please seek immediate medical care or contact your local emergency services.

Then continue with general educational information.

======================================================
INVALID INPUT
======================================================

If the user asks about movies, coding, sports, politics, vehicles or anything unrelated to healthcare, reply:

🩺 I can only assist with health and medical-related questions.

Please tell me your symptoms or ask a healthcare-related question.

======================================================
RESPONSE FORMAT
======================================================

🩺 Possible Condition

- ...

🔍 Why it may match

- ...

🛡️ Self-Care

- ...

🔄 Other Possible Conditions (if needed)

- ...

📌 Recommendation

- ...

======================================================

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
        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "answer": "Please enter a health-related question."
            })

        query = question.lower()

        # ==========================================
        # GREETINGS
        # ==========================================

        greetings = [
            "hi","hello","hey","hii","hiii",
            "good morning","good afternoon","good evening"
        ]

        if any(greet == query for greet in greetings):
            return jsonify({
                "answer": """
👋 Hello!

I'm **MediSense AI**, your AI Health Information Assistant.

I can help you with:

- Symptoms
- Diseases
- General Health Questions
- Preventive Care
- Healthy Lifestyle Tips

How can I help you today? 😊
"""
            })

        # ==========================================
        # THANK YOU
        # ==========================================

        thanks = [
            "thanks",
            "thank you",
            "thankyou",
            "thanks a lot",
            "thank you so much"
        ]

        if any(word == query for word in thanks):
            return jsonify({
                "answer": """
😊 You're welcome!

I'm glad I could help.

Take care and stay healthy! 💚
"""
            })

        # ==========================================
        # GOODBYE
        # ==========================================

        goodbye = [
            "bye",
            "goodbye",
            "see you",
            "bye bye",
            "take care"
        ]

        if any(word == query for word in goodbye):
            return jsonify({
                "answer": """
👋 Goodbye!

Take care of your health.

Have a wonderful day! 💙
"""
            })

        # ==========================================
        # ABOUT BOT
        # ==========================================

        about_bot = [
            "who are you",
            "what can you do",
            "help",
            "can you help me",
            "your name"
        ]

        if any(word == query for word in about_bot):
            return jsonify({
                "answer": """
🩺 I'm **MediSense AI**.

I can assist you with:

- Understanding symptoms
- General disease information
- Preventive healthcare
- Healthy lifestyle guidance
- Educational medical information

⚠ I do not replace a qualified doctor or provide confirmed diagnoses.
"""
            })

        # ==========================================
        # EMERGENCY KEYWORDS
        # ==========================================

        emergency_keywords = [
            "chest pain",
            "can't breathe",
            "cannot breathe",
            "difficulty breathing",
            "breathing difficulty",
            "unconscious",
            "stroke",
            "heart attack",
            "severe bleeding",
            "blood vomiting"
        ]

        emergency = any(word in query for word in emergency_keywords)

        # ==========================================
        # MEDICAL KEYWORDS
        # ==========================================

        medical_keywords = [

            "fever","cold","cough","headache","pain","vomiting",
            "nausea","dizziness","fatigue","weakness",
            "diarrhea","breathing","chest","throat",
            "sore","asthma","diabetes","sugar",
            "head","body","eye","ear","nose",
            "stomach","rash","infection","virus",
            "bacteria","covid","flu","dengue",
            "malaria","typhoid","migraine",
            "doctor","hospital","medicine",
            "health","medical","symptom","disease",
            "bp","pressure","heart","kidney",
            "liver","allergy","pregnancy",

            "bukhar","khansi","sardi","sir",
            "dard","ulti","chakkar","pet",
            "gala","saans","kamzori","jukaam"
        ]

        # ==========================================
        # INVALID INPUT
        # ==========================================

        if not any(word in query for word in medical_keywords):

            return jsonify({
                "answer": """
🩺 I can only assist with health and medical-related questions.

Examples:

- I have fever and headache
- Symptoms of diabetes
- Chest pain while breathing
- Bukhar aur sir dard

How can I help you today? 😊
"""
            })

        # ==========================================
        # RAG RESPONSE
        # ==========================================

        response = qa_chain.invoke(question)

        answer = response["result"]

        if emergency:

            answer = (
                "🚨 **This may require urgent medical attention.**\n\n"
                "Please seek immediate medical care or contact your local emergency services if your symptoms are severe.\n\n"
                + answer
            )

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
