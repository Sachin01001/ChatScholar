from flask import Flask, render_template, request, redirect
import os
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# === CONFIG ===
os.environ["OPENAI_API_KEY"] = "api_key" # add your api key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

app = Flask(__name__)
DATA_DIR = "__data__"
os.makedirs(DATA_DIR, exist_ok=True)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

# === UTILS ===
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def get_pdf_text(pdf_docs):
    combined_text = ""
    for pdf in pdf_docs:
        text = extract_text_from_pdf(pdf)
        combined_text += text
        file_path = os.path.join(DATA_DIR, pdf.filename + ".txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    return combined_text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vstore):
    llm = ChatOpenAI(model="deepseek/deepseek-r1:free", temperature=0.4)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vstore.as_retriever(), memory=memory)

def _grade_essay(essay):
    global rubric_text
    prompt = f"You are a professional essay evaluator. Grade this essay based on the rubric:\n\n{rubric_text}\n\nESSAY:\n{essay}"
    llm = ChatOpenAI(model="deepseek/deepseek-r1:free", temperature=0.4)
    response = llm.invoke(prompt)
    return response.content.replace("\n", "<br>")

# === ROUTES ===
@app.route('/')
def home():
    return render_template('new_home.html')

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('new_essay_rubric.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    global rubric_text
    result = None
    text = ""

    if request.method == 'POST':
        if request.form.get('essay_rubric'):
            rubric_text = request.form.get('essay_rubric')
        elif 'file' in request.files and request.files['file'].filename:
            text = extract_text_from_pdf(request.files['file'])
            result = _grade_essay(text)
        elif request.form.get('essay_text'):
            text = request.form.get('essay_text')
            result = _grade_essay(text)

    return render_template('new_essay_grading.html', result=result, input_text=text)

@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global chat_history
    if request.method == 'POST':
        user_question = request.form['user_question']
        result = conversation_chain({'question': user_question})
        answer = result['answer'].replace('\n', '<br>')
        chat_history.append((user_question, answer))
    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history = []
    return redirect('/chat')

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=False)
