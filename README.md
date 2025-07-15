# NLP RAG System

Build a system that answers questions on NLP topics using the Retrieval-Augmented Generation (RAG) approach, based on the Speech and Language Processing (Third Edition) book.

## Abstraction

The NLP Q&A system is an AI-powered application that streamlines searching within the Speech and Language Processing (Third Edition) book. It enables students, teachers, and researchers to quickly and easily retrieve relevant information from the book.

## Introduction

The NLP Q&A system is an AI-powered application that simplifies and enhances information retrieval using a Retrieval-Augmented Generation (RAG) approach. The system processes the main chapters of the Speech and Language Processing (Third Edition) book by breaking them into manageable chunks, generating embeddings, and storing them in a vector database to enable efficient similarity-based search. It then uses the RAG pipeline to generate accurate answers to user queries.

## Methodology

We follow a systematic process that begins with extracting text from the documents. The extracted text is then cleaned and preprocessed to prepare it for chunking. These chunks are converted into embeddings, which are then stored in a vector database for efficient retrieval during the question-answering process.

### Extract

In the extraction stage, we use the built-in PDF extractor from the LangChain library. After extracting the text, we apply a series of cleaning steps, including:

- Removing page headers  
- Removing copyright notices  
- Removing page numbers and repeated line breaks  
- Removing figure numbers  
- Normalizing whitespace

These steps ensure that the text is clean.

### Chunking

In the chunking stage, we use the built-in token-based chunking function from the LangChain library. We experiment with different chunk sizes,  100, 200, 300, and 384 tokens,  to evaluate which configuration yields the best performance and highest accuracy.

### Embeddings

In the embeddings stage, we use the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face to generate dense vector representations of the text chunks. These embeddings are later stored in a vector database for similarity-based retrieval.

### Vectorstore

In the vectorstore stage, we use the FAISS vector database for efficient similarity search and clustering of dense embeddings. This allows the system to quickly retrieve the most relevant chunks based on the user’s query.

### Pipeline

The pipeline involves testing the Mistral and Gemini models to identify the best-performing option. We use the built-in RAG pipeline from the LangChain library to integrate the retrieval and generation components effectively.

## Results

We evaluated the models by testing them with a set of questions. Our observations indicate that Mistral provides stronger and more accurate answers compared to Gemini. Additionally, when we adjusted the chunk size, we noticed that larger chunks allowed the models to access more context, resulting in better answers. However, even as the chunk size increased, the Mistral model consistently outperformed Gemini.

## Tech Stack

- Python
- LangChain
- gradio
- HuggingFace embeddings model `sentence-transformers/all-MiniLM-L6-v2`

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/amjadAwad95/nlp-rag-system.git
cd nlp-rag-system
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. Create a `.env` file in the root folder:

```file
GOOGLE_API_KEY = <GOOGLE_API_KEY>
MISTRAL_API_KEY = <MISTRAL_API_KEY>
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Run the app:

```bash
python main.py
```

## Conclusion

The NLP RAG system demonstrates how Retrieval-Augmented Generation (RAG) can enhance question-answering by effectively combining a large language model (LLM) with relevant retrieved information.

