from flask import Flask, render_template, request, jsonify 
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

app = Flask(__name__)

# Initialize RAG Pipeline
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatOllama(model="llama3", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3})
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():

    data = request.get_json() 
    user_question = data.get("question")
    
    if not user_question:
        return jsonify({"answer": "No question provided."}), 400

    # 2. Run the RAG workflow
    print(f"Querying Manual: {user_question}")
    response = qa_chain.invoke(user_question)
    
    # 3. Return a JSON response back to the JavaScript
    return jsonify({"answer": response['result']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)