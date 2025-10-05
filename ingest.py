from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


PDF_PATH = "Quran/quraan-english-arabic-mohsin-khan-and-al-hilali.pdf"

INDEX_DIR = "faiss_index"


print("📄 Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()


print("✂ Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"🔪 Total Chunks: {len(chunks)}")


print("🧬 Creating embeddings and saving to FAISS...")
embeddings = OpenAIEmbeddings(
model="text-embedding-3-large", # Better for English
openai_api_key=os.getenv("OPENAI_API_KEY")
)
db = FAISS.from_documents(chunks, embedding=embeddings)
db.save_local(INDEX_DIR)
print("✅ FAISS index saved!")