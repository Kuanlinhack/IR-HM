import argparse
import math
import os
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from tqdm import tqdm
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize variables
documents_dir = "data/EnglishNews"
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean and stem text, removing punctuation, stopwords, and applying stemming
def clean_and_stem_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.lower().split()  # Convert to lowercase and split into words
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return cleaned_words

# Compute term frequency (TF)
def compute_tf(words):
    tf = Counter(words)
    total_words = len(words)
    return {word: count / total_words for word, count in tf.items()}

# Compute inverse document frequency (IDF)
def compute_idf(documents):
    num_docs = len(documents)
    word_in_docs = defaultdict(int)
    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            word_in_docs[word] += 1
    return {word: math.log(num_docs / (1 + count)) for word, count in word_in_docs.items()}

# Load documents and compute TF-IDF
def load_documents_and_compute_tfidf(doc_directory):
    documents = []
    filenames = []
    tfidf_docs = []

    for file in tqdm(os.listdir(doc_directory), desc="Loading Documents"):
        if file.endswith(".txt"):
            with open(os.path.join(doc_directory, file), 'r') as f:
                content = clean_and_stem_text(f.read())
                documents.append(content)
                filenames.append(file)

    idf_scores = compute_idf(documents)

    for words in tqdm(documents, desc="Computing TF-IDF"):
        tf_scores = compute_tf(words)
        tfidf_scores = {word: tf_scores[word] * idf_scores[word] for word in tf_scores}
        tfidf_docs.append(tfidf_scores)

    return tfidf_docs, filenames, idf_scores, documents

# Process the query and compute similarity
def process_query(query, documents, filenames, tfidf_docs, idf_scores, weighting='TF-IDF', similarity_metric='Cosine'):
    query_words = clean_and_stem_text(query)

    if weighting == 'TF':
        query_tf = compute_tf(query_words)
        query_vector = query_tf
    else:
        query_tf = compute_tf(query_words)
        query_tfidf = {word: query_tf[word] * idf_scores.get(word, 0) for word in query_tf}
        query_vector = query_tfidf

    vocabulary = set(query_vector.keys()).union(*[set(doc.keys()) for doc in tfidf_docs])
    query_vector_np = np.array([query_vector.get(word, 0) for word in vocabulary])
    doc_vectors_np = [np.array([doc.get(word, 0) for word in vocabulary]) for doc in tfidf_docs]

    if similarity_metric == 'Cosine':
        similarities = cosine_similarity([query_vector_np], doc_vectors_np)[0]
    elif similarity_metric == 'Euclidean':
        similarities = euclidean_distances([query_vector_np], doc_vectors_np)[0]

    ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=(similarity_metric == 'Cosine'))
    return [(filenames[i], score) for i, score in ranked_docs[:10]]

# Format and print the results
def format_results(results, similarity_metric):
    print(f"{similarity_metric} Similarity")
    print(f"{'NewsID':<15}{'Score':<15}")
    print('-' * 30)
    for doc_id, score in results:
        print(f"{doc_id:<15} {score:.6f}")
    print()

# Main function to parse arguments and run the queries
def task_1_main():
    parser = argparse.ArgumentParser(description='Process English and Chinese queries.')
    parser.add_argument('--Eng_query', type=str, required=True, help='The English query to search in the documents.')
    parser.add_argument('--Chi_query', type=str, required=False, help='The Chinese query (not used in this script).')

    args = parser.parse_args()

    print('\nTask 1\n')
    # Load documents and compute TF-IDF
    tfidf_docs, filenames, idf_scores, raw_documents = load_documents_and_compute_tfidf(documents_dir)

    # Process the English query
    query = args.Eng_query

    print("TF + Cosine Similarity:")
    tf_cosine_results = process_query(query, raw_documents, filenames, tfidf_docs, idf_scores, 'TF', 'Cosine')
    format_results(tf_cosine_results, "TF")

    print("TF-IDF + Cosine Similarity:")
    tfidf_cosine_results = process_query(query, raw_documents, filenames, tfidf_docs, idf_scores, 'TF-IDF', 'Cosine')
    format_results(tfidf_cosine_results, "TF-IDF")

    print("TF + Euclidean Distance:")
    tf_euclidean_results = process_query(query, raw_documents, filenames, tfidf_docs, idf_scores, 'TF', 'Euclidean')
    format_results(tf_euclidean_results, "TF")

    print("TF-IDF + Euclidean Distance:")
    tfidf_euclidean_results = process_query(query, raw_documents, filenames, tfidf_docs, idf_scores, 'TF-IDF', 'Euclidean')
    format_results(tfidf_euclidean_results, "TF-IDF")

if __name__ == '__main__':
    task_1_main()

