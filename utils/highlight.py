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


def highlight_text(answer_text, source_docs, similarity_threshold=0.65):
    """
    compute the similarity between the generated text and the original text to the highlighted the generated text.
    :param answer_text: the generated text
    :param source_docs: the original texts
    :param similarity_threshold: the similarity threshold
    :return: highlighted text
    """
    all_source_sentences = []
    sentence_doc_map = []

    doc_names = []
    for doc in source_docs:
        name = doc.metadata.get("source", "Unknown source")
        doc_names.append(name)

    for doc_idx, doc in enumerate(source_docs):
        text = doc.page_content
        sentences = split_to_sentences(text)
        all_source_sentences.extend(sentences)
        sentence_doc_map.extend([doc_idx] * len(sentences))

    answer_sentences = split_to_sentences(answer_text)

    answer_emb = model.encode(answer_sentences, convert_to_tensor=True)
    source_emb = model.encode(all_source_sentences, convert_to_tensor=True)

    similarity_matrix = util.pytorch_cos_sim(answer_emb, source_emb)

    highlighted_answer = []

    for i, answer_sent in enumerate(answer_sentences):
        if re.fullmatch(r"^\d+\.?$", answer_sent):
            highlighted_answer.append(answer_sent)
            continue

        sim_row = similarity_matrix[i]
        above_threshold_indices = (sim_row >= similarity_threshold).nonzero(as_tuple=True)[0].tolist()

        if above_threshold_indices:
            matched_docs = set(sentence_doc_map[idx] for idx in above_threshold_indices)
            matched_doc_names = sorted(set(doc_names[doc_idx] for doc_idx in matched_docs))
            doc_label_str = ", ".join(matched_doc_names)

            # Add yellow background for the sentence and blue for the source
            highlighted_sentence = (
                f'<span style="background-color: red;">{answer_sent}</span> '
                f'<span style="color: #1976d2; font-size: 0.9em;">[Source: {doc_label_str}]</span>'
            )
            highlighted_answer.append(highlighted_sentence)
        else:
            highlighted_answer.append(answer_sent)

    return "<br><br>".join(highlighted_answer)
