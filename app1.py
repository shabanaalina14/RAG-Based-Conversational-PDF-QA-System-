import streamlit as st
import google.generativeai as genai
import PyPDF2
from chromadb import PersistentClient
import tempfile

# ---------- CONFIGURE GEMINI ----------
genai.configure(api_key="Your_API_KEY")
generation_model = genai.GenerativeModel("gemini-2.0-flash")

# ---------- LOAD PDF ----------
def load_pdf_text(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""
    return text

# ---------- SPLIT INTO CHUNKS ----------
def split_into_chunks(text, max_tokens=400):
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

# ---------- GET EMBEDDINGS ----------
def get_embeddings(chunks):
    embeddings, total = [], len(chunks)
    progress = st.progress(0)
    for i, chunk in enumerate(chunks):
        try:
            response = genai.embed_content(model="embedding-001", content=[chunk])
            embedding = response["embedding"][0]
        except Exception as e:
            st.warning(f"⚠️ Chunk {i} embedding failed: {e}")
            embedding = [0.0] * 768
        embeddings.append(embedding)
        progress.progress((i + 1) / total)
    return embeddings

# ---------- PROCESS PDF ----------
@st.cache_resource(show_spinner=False)
def process_pdf(uploaded_file):
    pdf_text = load_pdf_text(uploaded_file)
    if not pdf_text.strip():
        st.error("❌ No readable text found in the PDF.")
        st.stop()

    chunks = split_into_chunks(pdf_text)
    embeddings = get_embeddings(chunks)

    chroma_dir = tempfile.mkdtemp()
    client = PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="pdf_chunks")

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i]]
        )

    return collection

# ---------- ASK QUESTION WITH MEMORY ----------
def ask_question(query, collection):
    try:
        query_emb = genai.embed_content(model="embedding-001", content=[query])["embedding"][0]
    except Exception as e:
        return f"❌ Failed to embed question: {e}"

    try:
        results = collection.query(query_embeddings=[query_emb], n_results=3)
        docs = [doc for sublist in results["documents"] for doc in sublist]
        context = "\n".join(docs)

        # ---- Include chat history (memory) ----
        history = ""
        for turn in st.session_state.chat_history:
            history += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

        prompt = (
            "You are an assistant answering based only on the PDF context and conversation history.\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Context from PDF:\n{context}\n\n"
            f"Current Question: {query}\n"
            "Answer:"
        )

        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini generation failed: {e}"

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Gemini PDF QA", layout="wide")
st.title("📄 Gemini PDF QA Assistant with Memory")

st.markdown(
    """
    <style>
    .stTextInput input {border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("📄 Processing your PDF..."):
        try:
            collection = process_pdf(uploaded_file)
            st.success("✅ PDF processed and indexed! Ask your questions below 👇")
        except Exception as e:
            st.error(f"❌ Failed to process PDF: {e}")
            st.stop()

    # --- Display conversation in chat format ---
    for turn in st.session_state.chat_history:
        with st.chat_message("user", avatar="🤵"):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(turn["answer"])

    # --- Chat Input ---
    if prompt := st.chat_input("🔎 Ask a question about the PDF..."):
        with st.chat_message("user", avatar="🤵"):
            st.markdown(prompt)

        with st.spinner("✍️ Generating answer..."):
            answer = ask_question(prompt, collection)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)

        st.session_state.chat_history.append({"question": prompt, "answer": answer})

