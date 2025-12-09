

# 📄 RAG-Based-ConverstaionaL-PDF-QA-System 

An intelligent system that extracts insights from PDF documents through **multi-turn conversational QA with context memory**.  
Built with **Google Gemini, Streamlit, and ChromaDB** for real-time, retrieval-augmented responses.  

---

## 🚀 Features  
- 📑 **PDF Processing** – Upload and process PDFs into meaningful text chunks.  
- 🔍 **Semantic Search** – Generate embeddings with **Google Gemini** and store in **ChromaDB** for efficient retrieval.  
- 🧠 **Context Memory** – Maintain multi-turn conversation flow using `st.session_state`.  
- 🤖 **Conversational UI** – Real-time Q&A powered by **Gemini 2.0 Flash**.  

---

## 🛠️ Tech Stack  
- **LLM:** Google Gemini (Embeddings + Flash Model)  
- **Frontend:** Streamlit  
- **Database:** ChromaDB (Vector Store)  
- **Language:** Python  

---

## 📂 Project Structure  
```

├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── utils/              # Helper functions (chunking, embeddings, retrieval)
├── data/               # Sample PDFs
└── README.md           # Project documentation

````

---

## ⚡ How It Works  
1. **Upload PDF(s)** in the app.  
2. Text is **chunked & embedded** using Google Gemini embeddings.  
3. Embeddings are stored in **ChromaDB** for vector similarity search.  
4. User queries are answered via **retrieval-augmented generation (RAG)** with Gemini 2.0 Flash.  
5. **Conversation memory** ensures multi-turn contextual answers.  

---

## 📸 Demo  
(Add a screenshot or GIF of your Streamlit app here)

---

## 🔧 Installation  

1. Clone this repo:  
```bash
git clone https://github.com/your-username/gemini-pdf-qa.git
cd gemini-pdf-qa
````

2. Create a virtual environment & install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your **Google API key** to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

* Support for **multiple file formats** (DOCX, TXT).
* Advanced **memory mechanisms** (long-term storage).
* Deploy on **Streamlit Cloud / Hugging Face Spaces**.

---

## 🙌 Acknowledgements

* [Google Gemini](https://deepmind.google/technologies/gemini/)
* [Streamlit](https://streamlit.io/)
* [ChromaDB](https://www.trychroma.com/)
