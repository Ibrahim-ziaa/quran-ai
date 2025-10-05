import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="📖 Quran AI Companion", page_icon="🕌", layout="wide")
st.title("🕌 Quran AI Companion")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_retriever()

# Prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert Islamic scholar and Quran teacher. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Whenever a verse from the Quran is mentioned, put it in *italic* and wrap it in "quotation marks".
Respond with Quran-based explanation only, do not hallucinate. Always provide references with surah and verse number along with surah name!
""".strip()
)

# LLM and Chain
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Input box
user_question = st.text_input("Ask your question about the Quran and Hadith:")

if user_question:
    result = qa_chain.invoke({"query": user_question})
    quran_answer = result["result"]

    # Collect relevant Hadiths and Quranic verses
    hadiths = []
    verses = []
    for doc in result["source_documents"]:
        content = doc.page_content.strip()
        if "hadith" in content.lower():
            hadiths.append(content)
        else:
            verses.append(content)

    # Layout
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### 🧠 Answer")
        st.markdown("**Explanation:**")
        st.success(quran_answer)

    with right:
        st.markdown("### 📖 Quranic Verses & Hadiths")
        if verses:
            st.markdown("#### 📘 Quranic Verses")
            for i, v in enumerate(verses, 1):
                st.markdown(f"**Verse {i}:** *\"{v}\"*")

        if hadiths:
            st.markdown("#### 📜 Hadiths")
            for i, h in enumerate(hadiths, 1):
                st.markdown(f"**Hadith {i}:** *\"{h}\"*")

        if not verses and not hadiths:
            st.info("No Quranic verses or hadiths found in this response.")
