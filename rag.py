import os
import gradio as gr
import warnings
from typing import List
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

class PDFChatbot:
    def __init__(self, pdf_path):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.initialize_pdf(pdf_path)
        
    def load_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return []

    def split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            return Chroma.from_documents(documents, embeddings)
        except Exception as e:
            print(f"Error creating vector: {e}")
            return None

    def create_qa_chain(self, vector_store):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest", 
                temperature=0.1, 
                convert_system_message_to_human=True,
                model_kwargs={
                    "max_output_tokens": 8192,
                    "top_k": 10,
                    "top_p": 0.95
                }
            )
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            return None
        
    def process_pdf_query(self, query: str, retrieve_source_documents=False) -> str:
        if not self.vector_store or not self.qa_chain:
            return "Error: Vector store or QA chain not initialized. Please load PDFs first."

        try:
            response = self.qa_chain.invoke({"query": query})
            top_chunks = "\n\n".join([
                f"Chunk (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content[:500]}"
                for doc in response['source_documents']
            ])

            if retrieve_source_documents:
                return (f"{response['result']}\n\nTop Relevant Chunks:\n{top_chunks}")

            else:
                return (f"{response['result']}")

        except Exception as e:
            return f"Error processing query: {e}"

    def initialize_pdf(self, pdf_path):
        documents = self.load_pdf(pdf_path)
        if not documents:
            raise ValueError("Error: Could not load PDF")

        split_docs = self.split_documents(documents)
        self.vector_store = self.create_vector_store(split_docs)
        if not self.vector_store:
            raise ValueError("Error: Could not create vector store")

        self.qa_chain = self.create_qa_chain(self.vector_store)
        if not self.qa_chain:
            raise ValueError("Error: Could not create QA chain")


def launch_gradio(pdf_path, retrieve_source_documents):
    """Launch Gradio interface for PDF Q&A."""
    chatbot = PDFChatbot(pdf_path)

    with gr.Blocks() as demo:
        gr.Markdown("## 📄Training Reproducible DL Models Q&A Chatbot")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Ask a Question")
                submit_btn = gr.Button("Ask Question")
                output = gr.Textbox(label="Response", lines=10)
        
        def process_query(query):
            return chatbot.process_pdf_query(query, retrieve_source_documents)
        
        def clear_input():
            return gr.update(value="")  # Instantly clear the input field
        
        # Set up interactions
        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[output]
        )

        submit_btn.click(
            fn=clear_input,
            inputs=[],
            outputs=[query_input]
        )

        query_input.submit(
            fn=process_query,
            inputs=[query_input],
            outputs=[output]
        )

        query_input.submit(
            fn=clear_input,
            inputs=[],
            outputs=[query_input]
        )

    demo.launch()

def main(pdf_path="./rep_dl.pdf", retrieve_source_documents=False):
    launch_gradio(pdf_path, retrieve_source_documents)

if __name__ == "__main__":
    main()
