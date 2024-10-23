import math
import os
import numpy as np
import nltk
from collections import Counter, defaultdict
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from VectorSpace import VectorSpace  # Import the VectorSpace class

# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean and stem text
def clean_and_stem_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)  # Join words back into a string

# Load documents
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    return clean_and_stem_text(content)

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

def cosine_similarity(queries_vectors, docs_vectors):
    similarities = np.array([[np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d)) for d in docs_vectors] for q in queries_vectors])
    print("Shape of similarity matrix:", similarities.shape)
    print("Sample similarities:", similarities[0][:10])
    return similarities

def inner_product(queries_vectors, docs_vectors):
    return np.array([[np.dot(q, d) for d in docs_vectors] for q in queries_vectors])

def euclidean_distance(queries_vectors, docs_vectors):
    return np.array([[np.linalg.norm(q - d) for d in docs_vectors] for q in queries_vectors])

# Rank top K documents
def rank_top_k(similarity_matrix, K=10):
    return np.argsort(-similarity_matrix, axis=1)[:, :K]


# Compute Recall@10
def compute_recall_at_10(queries, top_k_results, relevance_dict, collections):
    query_id = list(queries.keys())[0]
    query = queries[query_id]
    
    print(f"Query ID: {query_id}")
    print(f"Query: {query}")
    
    relevant_docs = set(relevance_dict[query_id])
    print(f"Relevant docs: {relevant_docs}")
    
    print(f"Top 10 results indices: {top_k_results[0]}")
    
    retrieved_docs = set([doc_id + 1 for doc_id in top_k_results[0]])
    print(f"Retrieved docs: {retrieved_docs}")
    
    # 打印相關文檔的內容
    print("Content of relevant docs:")
    for doc_id in list(relevant_docs)[:3]:  # 只打印前3個相關文檔
        print(f"Doc {doc_id}: {collections[f'd{doc_id}'][:100]}...")  # 只打印前100個字符
    
    # 打印檢索到的文檔的內容
    print("Content of retrieved docs:")
    for doc_id in list(retrieved_docs)[:3]:  # 只打印前3個檢索到的文檔
        print(f"Doc {doc_id}: {collections[f'd{doc_id}'][:100]}...")  # 只打印前100個字符
    
    intersection = relevant_docs.intersection(retrieved_docs)
    print(f"Intersection: {intersection}")
    
    recall = len(intersection) / len(relevant_docs) if relevant_docs else 0
    print(f"Recall: {recall}")

    retrieved_docs = set([int(collections_files[doc_id].split('d')[1]) for doc_id in top_k_results[0]])
    print(f"Retrieved docs: {retrieved_docs}")

    
    return recall

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
    # print(f"queries[:5]: {list(queries.values())[:5]}")
    # print(f"collections[:5]: {list(collections.values())[:5]}")

    relevance_dict = load_relevance(rel_file_path)
    # print(f"relevance_dict[:5]: {list(relevance_dict.values())[:5]}")
    
    # Define collections_files
    collections_files = [f"d{doc_id}" for doc_id in range(1, len(collections) + 1)]

    # Use VectorSpace instead of TfidfVectorizer
    all_documents = list(queries.values()) + list(collections.values())
    vector_space = VectorSpace(all_documents)
    
    # Get document vectors
    all_vectors = vector_space.documentVectors
    queries_vectors = np.array(all_vectors[:len(queries)])
    docs_vectors = np.array(all_vectors[len(queries):])

    # Calculate similarities
    cosine_sim = cosine_similarity(queries_vectors, docs_vectors)
    inner_prod_sim = inner_product(queries_vectors, docs_vectors)
    euclidean_sim = euclidean_distance(queries_vectors, docs_vectors)
    print("Sample query vector:", queries_vectors[0][:10])
    print("Sample document vector:", docs_vectors[0][:10])
    print("Sample cosine similarity:", cosine_sim[0][:10])

    # 檢查文檔 ID 的映射
    print("First 10 document IDs:", collections_files[:10])

    # Rank top 10 documents for each metric
    top_k_cosine = rank_top_k(cosine_sim, K=10)
    print(f"top_k_cosine[:5]: {top_k_cosine[:5]}")
    top_k_inner = rank_top_k(inner_prod_sim, K=10)
    top_k_euclidean = rank_top_k(-euclidean_sim, K=10)  # Negative to get top K for distance

    # Calculate metrics for Cosine Similarity
    print("Metrics for Cosine Similarity")
    print(f"Recall@10: {compute_recall_at_10(queries, top_k_cosine, relevance_dict, collections)}")
    print(f"MAP@10: {compute_map_at_10(queries, top_k_cosine, relevance_dict, collections_files)}")
    print(f"MRR@10: {compute_mrr_at_10(queries, top_k_cosine, relevance_dict, collections_files)}")

    # Similarly, calculate metrics for Inner Product and Euclidean Distance
    # ...

    print(f"Length of collections: {len(collections)}")
    print(f"Length of collections_files: {len(collections_files)}")

if __name__ == "__main__":
    main()
