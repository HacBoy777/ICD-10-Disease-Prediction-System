import spacy


nlp = spacy.load("en_core_web_sm")


def extract_phrases(paragraph):

    doc = nlp(paragraph)

    phrases = []


    # Noun chunks

    for chunk in doc.noun_chunks:

        text = chunk.text.strip().lower()

        if len(text.split()) >= 2:

            if text not in phrases:
                phrases.append(text)


    # Adjective + noun

    for token in doc:

        if token.pos_ == "ADJ":

            for child in token.children:

                if child.pos_ == "NOUN":

                    phrase = (
                        token.text
                        + " "
                        + child.text
                    ).lower()

                    if phrase not in phrases:
                        phrases.append(phrase)


    # Sentences

    for sentence in doc.sents:

        text = sentence.text.strip().lower()

        if len(text.split()) >= 2:

            if text not in phrases:
                phrases.append(text)


    return list(dict.fromkeys(phrases))