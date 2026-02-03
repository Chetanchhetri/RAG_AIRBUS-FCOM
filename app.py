from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings # Updated import
from langchain_community.vectorstores import FAISS

# 1. Load your PDF (Ensure the file name is exactly as it appears in your folder)
loader = PyPDFLoader("A320_data.pdf")
documents = loader.load()

# 2. Chunk the text for the vector DB
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3. Create Local Embeddings (requires 'ollama pull nomic-embed-text' first)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Build and Save FAISS DB locally
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("faiss_index")
print("Local Llama 3 Vector DB created successfully!")
