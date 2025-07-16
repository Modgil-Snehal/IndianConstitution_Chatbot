from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm
import time

# Load and split
loader = TextLoader("indian_constitution.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Initialize embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- Embed with progress bar ---
print(f"🔄 Starting embedding of {len(chunks)} chunks...")

start_time = time.time()
chunk_texts = [chunk.page_content for chunk in chunks]
embedded_vectors = []

for i in tqdm(range(len(chunk_texts)), desc="🔎 Embedding chunks"):
    chunk = chunk_texts[i]
    embedded_vectors.append(embeddings.embed_query(chunk))

embedding_time = time.time() - start_time
print(f"✅ Finished embedding in {embedding_time:.2f} seconds ({embedding_time / len(chunk_texts):.2f} sec/chunk)")

# Build FAISS vectorstore manually
vectorstore = FAISS.from_embeddings(embedded_vectors, chunk_texts)

# Save FAISS to disk
save_start = time.time()
vectorstore.save_local("faiss_index")
print(f"💾 FAISS saved in {time.time() - save_start:.2f} seconds")
