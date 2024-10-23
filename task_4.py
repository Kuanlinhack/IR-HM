import math
import os
import numpy as np
import nltk
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean and stem text
def clean_and_stem_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return cleaned_words

# Load documents
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    return " ".join(clean_and_stem_text(content))

# Load queries and collections
def load_data(queries_folder, collections_folder):
    queries = {}
    for query_file in os.listdir(queries_folder):
        query_id = query_file.split(".")[0]
        queries[query_id] = load_document(os.path.join(queries_folder, query_file))
    
    collections = {}
    for doc_file in os.listdir(collections_folder):
        doc_id = doc_file.split(".")[0]
        collections[doc_id] = load_document(os.path.join(collections_folder, doc_file))
    
    return queries, collections

# Load relevance info
def load_relevance(rel_file_path):
    relevance_dict = {}
    with open(rel_file_path, 'r') as file:
        for line in file:
            query_id, relevant_docs = line.strip().split('\t')
            relevance_dict[query_id] = eval(relevant_docs)
    return relevance_dict

def cosine_distance(queries_tfidf, docs_tfidf):
    return 1 - pairwise_distances(queries_tfidf, docs_tfidf, metric='cosine')

def inner_product(queries_tfidf, docs_tfidf):
    # Use .dot() for sparse matrices and convert to dense
    return queries_tfidf.dot(docs_tfidf.T).toarray()

def euclidean_distance(queries_tfidf, docs_tfidf):
    return pairwise_distances(queries_tfidf, docs_tfidf, metric='euclidean')


# Rank top K documents
def rank_top_k(similarity_matrix, K=10):
    if len(similarity_matrix.shape) == 2:
        return np.argsort(-similarity_matrix, axis=1)[:, :K]
    else:
        raise ValueError("Expected 2D similarity matrix, but got {}".format(similarity_matrix.shape))

# Compute Recall@10
def compute_recall_at_10(queries, top_k, relevance_dict, collections_files):
    recall_sum = 0.0
    for i, query_id in enumerate(queries.keys()):
        relevant_docs = relevance_dict.get(query_id, [])
        top_k_docs = [collections_files[idx].split('.')[0] for idx in top_k[i]]
        relevant_found = len(set(relevant_docs).intersection(set(top_k_docs)))
        recall_sum += relevant_found / len(relevant_docs) if relevant_docs else 0
    return recall_sum / len(queries)

# Compute MAP@10
def compute_map_at_10(queries, top_k, relevance_dict, collections_files):
    average_precisions = []
    for i, query_id in enumerate(queries.keys()):
        relevant_docs = relevance_dict.get(query_id, [])
        relevant_found = 0
        precision_at_k = []
        top_k_docs = [collections_files[idx].split('.')[0] for idx in top_k[i]]
        for k, doc_id in enumerate(top_k_docs, start=1):
            if doc_id in relevant_docs:
                relevant_found += 1
                precision_at_k.append(relevant_found / k)
        if precision_at_k:
            average_precisions.append(np.mean(precision_at_k))
        else:
            average_precisions.append(0)
    return np.mean(average_precisions)

# Compute MRR@10
def compute_mrr_at_10(queries, top_k, relevance_dict, collections_files):
    mrr = 0.0
    for i, query_id in enumerate(queries.keys()):
        relevant_docs = relevance_dict.get(query_id, [])
        top_k_docs = [collections_files[idx].split('.')[0] for idx in top_k[i]]
        for rank, doc_id in enumerate(top_k_docs, start=1):
            if doc_id in relevant_docs:
                mrr += 1.0 / rank
                break
    return mrr / len(queries)

# Main function
def main():
    # Load queries and collections
    queries_folder = 'data/smaller_dataset/queries'
    collections_folder = 'data/smaller_dataset/collections'
    rel_file_path = 'data/smaller_dataset/rel.tsv'
    queries, collections = load_data(queries_folder, collections_folder)
    print(f"queries[:5]: {list(queries.values())[:5]}")
    print(f"collections[:5]: {list(collections.values())[:5]}")

 
    relevance_dict = load_relevance(rel_file_path)
    print(f"relevance_dict[:5]: {list(relevance_dict.values())[:5]}")
    # Define collections_files
    collections_files = list(collections.keys())

    # Convert queries and documents to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    all_texts = list(queries.values()) + list(collections.values())
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split into query and document vectors
    queries_tfidf = tfidf_matrix[:len(queries)]
    docs_tfidf = tfidf_matrix[len(queries):]

    # Calculate similarities
    cosine_sim = cosine_distance(queries_tfidf, docs_tfidf)
    inner_prod_sim = inner_product(queries_tfidf, docs_tfidf)
    euclidean_sim = euclidean_distance(queries_tfidf, docs_tfidf)

    # Rank top 10 documents for each metric
    top_k_cosine = rank_top_k(cosine_sim, K=10)
    print(f"top_k_cosine[:5]: {top_k_cosine[:5]}")
    top_k_inner = rank_top_k(inner_prod_sim, K=10)
    top_k_euclidean = rank_top_k(-euclidean_sim, K=10)  # Negative to get top K for distance

    # Calculate metrics for Cosine Similarity
    print("Metrics for Cosine Similarity")
    print(f"Recall@10: {compute_recall_at_10(queries, top_k_cosine, relevance_dict, collections_files)}")
    print(f"MAP@10: {compute_map_at_10(queries, top_k_cosine, relevance_dict, collections_files)}")
    print(f"MRR@10: {compute_mrr_at_10(queries, top_k_cosine, relevance_dict, collections_files)}")

    # Similarly, calculate metrics for Inner Product and Euclidean Distance
    # ...

if __name__ == "__main__":
    main()
