📄 Gemini-RAG: PDF Q&A Chatbot
Gemini-RAG is a Python-based application that enables users to interactively query PDF documents using Retrieval-Augmented Generation (RAG) techniques. It uses Google Generative AI models and LangChain to process PDF files, create vector embeddings, and respond to user queries via a Gradio interface.

🚀 Features
PDF Document Processing
Load, split, and process PDF documents into retrievable chunks for efficient querying.

Embedding and Retrieval
Creates embeddings using GoogleGenerativeAIEmbeddings and stores them in a Chroma vector database for efficient similarity-based document retrieval.

Interactive Q&A
Query the document using a chatbot powered by Google Generative AI (Gemini) and retrieve relevant information.

Gradio Interface
A user-friendly Gradio-based UI for real-time question-answering.

🛠️ Technologies Used
LangChain: For managing document loading, splitting, and retrieval QA chains.
Google Generative AI: To generate embeddings and handle the LLM-based chatbot.
Chroma: As a vector database for efficient document similarity search.
Gradio: For building an interactive web-based user interface.
📥 Installation
Prerequisites
Python 3.8 or higher
A Google Cloud account with access to Google Generative AI (for embedding and LLM models)
Install required Python packages using pip:
bash
Copy code
pip install -r requirements.txt
Environment Setup
Create a .env file in the project directory with your Google Generative AI API Key:
makefile
Copy code
GOOGLE_API_KEY=<your_api_key>
▶️ Usage
Place the PDF document you want to query in the project directory (e.g., rep_dl.pdf).
Run the chatbot:
bash
Copy code
python rag.py
Open the Gradio interface in your browser and start asking questions about the document.
Optional Parameters
pdf_path: Path to the PDF file (default: ./rep_dl.pdf).
retrieve_source_documents: Set to True to retrieve the most relevant chunks along with the response (default: False).
Example:

bash
Copy code
python rag.py --pdf_path ./your_file.pdf --retrieve_source_documents True
📂 Project Structure
bash
Copy code
Gemini-RAG/
│
├── rag.py                     # Main script for chatbot functionality
├── requirements.txt           # Required Python dependencies
├── .env                       # API keys (to be created)
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
🖼️ Screenshots
Gradio Interface
Coming soon!

🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📧 Contact
For any questions or suggestions, please reach out to Nour Ayman via GitHub or email.
