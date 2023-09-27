import math
import os
import string

import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = { }
    for fileName in os.listdir(directory):
        filepath = os.path.join(directory, fileName)
        with open(filepath, 'r') as file_reader:
            files[fileName] = file_reader.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized = nltk.tokenize.word_tokenize(document.lower())
    words = []
    for word in tokenized:
        if word not in nltk.corpus.stopwords.words("english") and word not in string.punctuation:
            words.append(word)
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_vals = { }

    words = []
    for doc in documents.values():
        words += doc
    words = set(words)
    for word in words:
        word_count = sum([1 for doc_words in documents.values() if word in doc_words])
        idf_vals[word] = math.log(len(documents) / word_count)
    return idf_vals


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    top = { }
    for fileName, file_text in files.items():
        val = sum([file_text.count(word) * idfs[word] for word in query if word in file_text])
        if val != 0:
            top[fileName] = val

    return [fileName for fileName, val in sorted(top.items(), key=lambda item: item[1], reverse=True)][:n:]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top = { }
    for sentence, s_words in sentences.items():
        val = sum(idfs[word] for word in query if word in s_words)
        if val != 0:
            query_term_density = sum([s_words.count(word) for word in query if word in sentence]) / len(query)
            top[sentence] = (val, query_term_density)

    return [sentence for sentence, val in sorted(top.items(), key=lambda item: item[1], reverse=True)][:n:]


if __name__ == "__main__":
    main()
