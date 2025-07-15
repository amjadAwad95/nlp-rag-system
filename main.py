import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from utils import clean
from dotenv import load_dotenv

load_dotenv()

dir = "documents"
documents_path = os.listdir(dir)
documents_path = [f"{dir}/{file}" for file in documents_path]

documents = []
for file in documents_path:
    loader = PyPDFLoader(file)
    loaded_docs = loader.load()

    for doc in loaded_docs:
        doc.page_content = clean(doc.page_content)

    documents.extend(loaded_docs)

text_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)

retriever  = vectorstore.as_retriever()
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0.8, max_retries=2)

rag_pipeline = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def generate(query):
    response = rag_pipeline.invoke(query)

    return response["result"]


css = """
    .gradio-container {
    max-width: 800px;
    margin: 0 auto;
    }
    section {
    margin-bottom: 2em !important;
    }
    .markdown-body h3 {
    margin-top: 1.5em;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.3em;
    }
"""

with gr.Blocks(title="NLP Q&A System", css=css) as demo:
    gr.Markdown("""
    # <center>NLP Q&A System</center>
    *<center>Speech and Language Processing - Third Edition</center>*
    """)

    gr.Markdown("### 1. Enter Your Query")
    question = gr.Textbox(
        label="",
        placeholder="Ask about attention mechanisms, training protocols, or architectural variants...",
        lines=3,
        max_lines=6
    )

    with gr.Group():
        submit_btn = gr.Button("Generate Analysis", variant="primary")
        clear_btn = gr.Button("Clear", size="sm")

    gr.Markdown("### 2. Generated Analysis")
    answer = gr.Markdown(
        value="*Your detailed analysis will appear here...*",
        elem_classes="markdown-body"
    )

    gr.Markdown("### 3. Example Queries")
    gr.Examples(
        examples=[
            "Explain the quadratic complexity problem in attention",
            "Compare encoder-only and decoder-only architectures",
            "How does transformers work?"
        ],
        inputs=question,
        label="Try these examples →"
    )

    gr.Markdown("---")
    gr.Markdown("*Academic use only - Generated content may require verification*")

    submit_btn.click(
        fn=generate,
        inputs=question,
        outputs=answer
    )

    clear_btn.click(
        fn=lambda: ("", "*Cleared - Enter a new query above*"),
        inputs=None,
        outputs=[question, answer]
    )

demo.launch(
    debug=True,
    share=False,
    server_name="0.0.0.0",
    server_port=7860
)