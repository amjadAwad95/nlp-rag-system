import re

def clean(text):
    """
    Clean the text by remove page headers like, remove copyright notes, remove page numbers and repeated line breaks,
    remove figure numbers like, and normalize whitespace
    :param text: the text to be cleaned
    :return: the cleaned text
    """
    text = re.sub(r'CHAPTER\s+\d+\s+.*?\n', '', text, flags=re.IGNORECASE)

    text = re.sub(r'Copyright\s+©\s+\d{4}.*?\n', '', text, flags=re.IGNORECASE)

    text = re.sub(r'Page\s*\d+|^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{2,}', '\n', text)

    text = re.sub(r'Figure\s+\d+\.\d+', '', text)

    text = re.sub(r'(Figure|Image)\s*\d+(\.\d+)?', '', text, flags=re.IGNORECASE)

    text = re.sub(r'(Figure|Image):.*', '', text, flags=re.IGNORECASE)

    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    text = re.sub(r'\[Image.*?\]', '', text, flags=re.IGNORECASE)


    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()
