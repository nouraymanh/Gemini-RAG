# 📄 Gemini-RAG: PDF Q&A Chatbot

Gemini-RAG is a Python-based application that enables users to interactively query PDF documents using **Retrieval-Augmented Generation (RAG)** techniques. It leverages Google Generative AI models and LangChain to process PDF files, create vector embeddings, and respond to user queries via a Gradio interface.

---

## 🚀 Features

- **PDF Document Processing**  
  Load, split, and process PDF documents into retrievable chunks for efficient querying.
  
- **Embedding and Retrieval**  
  Creates embeddings using **GoogleGenerativeAIEmbeddings** and stores them in a **Chroma** vector database for efficient similarity-based document retrieval.

- **Interactive Q&A**  
  Query the document using a chatbot powered by **Google Generative AI (Gemini)** and retrieve relevant information.

- **Gradio Interface**  
  A user-friendly Gradio-based UI for real-time question-answering.

---

## 🛠️ Technologies Used

- **[LangChain](https://www.langchain.com/)**: For managing document loading, splitting, and retrieval QA chains.  
- **[Google Generative AI](https://cloud.google.com/generative-ai/)**: To generate embeddings and handle the LLM-based chatbot.  
- **[Chroma](https://www.trychroma.com/)**: As a vector database for efficient document similarity search.  
- **[Gradio](https://gradio.app/)**: For building an interactive web-based user interface.

---

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- A Google Cloud account with access to **Google Generative AI** (for embedding and LLM models)
- Install required Python packages using `pip`:
  ```bash
  pip install -r requirements.txt
  ```

### Environment Setup

- Create a `.env` file in the project directory with your **Google Generative AI API Key**:
  ```
  GOOGLE_API_KEY=<your_api_key>
  ```

---

## ▶️ Usage

1. Place the PDF document you want to query in the project directory (e.g., `rep_dl.pdf`).
2. Run the chatbot:
   ```bash
   python rag.py
   ```
3. Open the Gradio interface in your browser and start asking questions about the document.

### Optional Parameters

- `pdf_path`: Path to the PDF file (default: `./rep_dl.pdf`).  
- `retrieve_source_documents`: Set to `True` to retrieve the most relevant chunks along with the response (default: `False`).

Example:
```bash
python rag.py --pdf_path ./your_file.pdf --retrieve_source_documents True
```

---

## 📂 Project Structure

```
Gemini-RAG/
│
├── rag.py                     # Main script for chatbot functionality
├── requirements.txt           # Required Python dependencies
├── .env                       # API keys (to be created)
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.
