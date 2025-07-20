import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def split_to_sentences(text):
    """
    Take a sentence and split it into a list of sentences.
    :param text: text to split
    :return: a list of sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    clean_sentences = []
    buffer = ""

    for sent in sentences:
        if re.fullmatch(r"\d+\.?", sent.strip()):
            if clean_sentences:
                clean_sentences[-1] += " " + sent.strip()
            else:
                buffer += sent.strip()
        else:
            clean_sentences.append(sent.strip())

    return clean_sentences


def highlight_documents(source_docs, answer_text, similarity_threshold=0.70):
    """
    Highlight the source documents according to the similarity threshold.
    :param source_docs: source documents
    :param answer_text: generated text
    :param similarity_threshold: similarity threshold
    :return: the source documents with highlighted text
    """
    sentence_docs = [split_to_sentences(doc.page_content) for doc in source_docs]

    all_sentences = [sent for doc in sentence_docs for sent in doc]
    sentence_embeddings = model.encode(all_sentences, convert_to_tensor=True)

    sentence_encodings = []
    idx = 0
    for doc in sentence_docs:
        n = len(doc)
        sentence_encodings.append(sentence_embeddings[idx:idx + n])
        idx += n

    answer_sentences = split_to_sentences(answer_text)
    answer_embeddings = model.encode(answer_sentences, convert_to_tensor=True)

    highlighted_output = []

    for doc_sentences, doc_embeddings in zip(sentence_docs, sentence_encodings):
        sim_matrix = util.pytorch_cos_sim(doc_embeddings, answer_embeddings)
        max_similarities = sim_matrix.max(dim=1).values  # Best similarity per sentence

        highlighted_sentences = []
        for sentence, sim_score in zip(doc_sentences, max_similarities):
            if sim_score.item() >= similarity_threshold:
                highlighted_sentences.append(f'<span style="background-color: red;">{sentence}</span>')
            else:
                highlighted_sentences.append(sentence)

        highlighted_output.append(" ".join(highlighted_sentences))

    return highlighted_output