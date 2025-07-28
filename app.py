import gradio as gr
from dotenv import load_dotenv
from pipeline import MemoryRAGPipeline
from utils import highlight_documents, load_uploaded_documents
from langdetect import detect
from arabic_config import arabic_pipeline_config

load_dotenv()

rag = MemoryRAGPipeline(["NLP"])
rag.setup_pipeline()

uploaded_rag_pipeline = {"rag": None, "rtl":False}

def generate(query):
    raw_answer, source_docs = rag.answer(query)
    highlighted_docs = highlight_documents(source_docs, raw_answer)
    highlighted_html = "<hr>".join([f"<div>{doc}</div>" for doc in highlighted_docs])
    return raw_answer, highlighted_html

def summarize_answer(query):
    raw_answer, _ = rag.answer(query)
    summary = rag.summarize(raw_answer)
    return summary


def setup_user_rag(files):
    if not files:
        return "File not found"
    documents = load_uploaded_documents(files)

    text = ""

    for doc in documents:
        text += doc.page_content

    main_language = detect(text)

    user_rag = MemoryRAGPipeline(documents)
    if main_language == "ar":
        user_rag.setup_pipeline(config=arabic_pipeline_config)
        uploaded_rag_pipeline["rtl"] = True
    else:
        user_rag.setup_pipeline()
        uploaded_rag_pipeline["rtl"] = False

    uploaded_rag_pipeline["rag"] = user_rag

    return "Ready to chat"


def user_rag_generate(query):
    user_rag = uploaded_rag_pipeline["rag"]

    if not uploaded_rag_pipeline["rag"]:
        return gr.update(value="*Please upload PDFs first.*")

    raw_answer, source_docs = user_rag.answer(query)

    if uploaded_rag_pipeline["rtl"]:
        wrapped = f'<div dir="rtl" style="text-align: right;">{raw_answer}</div>'
    else:
        wrapped = raw_answer

    return wrapped

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
    span.source {
        color: #1976d2;
        font-size: 0.9em;
        margin-left: 8px;
    }
    span.highlight {
        background-color: #fff59d;
    }
"""

with gr.Blocks(title="NLP Q&A System", css=css) as demo:
    gr.Markdown("""
        # <center>NLP Q&A System</center>
        *<center>Speech and Language Processing - Third Edition</center>*
    """)

    with gr.Tabs():
        with gr.Tab("Ask a Question"):
            gr.Markdown("### 1. Enter Your Query")
            question = gr.Textbox(
                label="",
                placeholder="Ask about attention mechanisms, training protocols, or architectural variants...",
                lines=3,
                max_lines=6
            )

            with gr.Group():
                submit_btn = gr.Button("Generate Answer", variant="primary")
                clear_btn = gr.Button("Clear", size="sm")

            gr.Markdown("### 2. Generated Answer")
            answer = gr.Markdown(
                value="*Your detailed answer will appear here...*",
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

        with gr.Tab("Highlighted Documents"):
            gr.Markdown("### Highlighted Sentences from Source Documents")
            highlighted_answer = gr.HTML(
                value="<i>Highlighted source information will appear here...</i>"
            )

        with gr.Tab("Summarization"):
            gr.Markdown("### Summarize Answer")
            summarize_btn = gr.Button("Summarize Answer")
            summary_output = gr.Markdown(value="*Summary will appear here...*")

        with gr.Tab("Chat with Your Files"):
            gr.Markdown("### Upload Your PDFs and Ask Questions")

            uploaded_files = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF Files")

            upload_status = gr.Markdown("*No files uploaded yet*")

            uploaded_question = gr.Textbox(
                placeholder="Ask a question about your uploaded documents...",
                lines=3
            )

            uploaded_answer = gr.Markdown(value="*Answer will appear here...*")
            uploaded_submit = gr.Button("Generate Answer")
            clear_user_btn = gr.Button("Clear", size="sm")

            uploaded_files.change(
                fn=setup_user_rag,
                inputs=uploaded_files,
                outputs=upload_status
            )

            uploaded_submit.click(
                fn=user_rag_generate,
                inputs=uploaded_question,
                outputs=uploaded_answer
            )

            clear_user_btn.click(
                fn=lambda: ("", "*Cleared - Enter a new query above*"),
                inputs=None,
                outputs=[uploaded_question, uploaded_answer]
            )

    submit_btn.click(
        fn=generate,
        inputs=question,
        outputs=[answer, highlighted_answer]
    )

    summarize_btn.click(
        fn=summarize_answer,
        inputs=question,
        outputs=summary_output
    )

    clear_btn.click(
        fn=lambda: ("", "*Cleared - Enter a new query above*", "*Cleared*", "*Cleared*"),
        inputs=None,
        outputs=[question, answer, highlighted_answer, summary_output]
    )

demo.launch(
    debug=True,
    share=False,
    server_name="0.0.0.0",
    server_port=7860
)
