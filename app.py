import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import faiss
import nltk

# ==============================
# FIX NLTK (HUGGINGFACE SAFE)
# ==============================
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Multi Search Engine", layout="wide")
st.title("🔍 Advanced Multi-Search Product Engine")

# ==============================
# LOAD MODEL (NO CACHE BUG)
# ==============================
if "model" not in st.session_state:
    with st.spinner("Loading AI model..."):
        st.session_state.model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu'
        )

model = st.session_state.model

# ==============================
# SEARCH INFO
# ==============================
search_info = {
    "Keyword": ("Find exact word match", "iphone → iPhone"),
    "Regex": ("Pattern-based search", "^S → Samsung"),
    "Boolean": ("Use AND / OR", "nike AND shoes"),
    "Fuzzy": ("Handles spelling mistakes", "iphon → iPhone"),
    "N-Gram": ("Partial word match", "iph → iPhone"),
    "Prefix": ("Starts with query", "app → Apple"),
    "Suffix": ("Ends with query", "laptop → Dell Laptop"),
    "TF-IDF": ("Ranks important words", "wireless headphones"),
    "BM25": ("Advanced keyword ranking", "gaming laptop"),
    "Semantic": ("Understands meaning", "sports footwear"),
    "FAISS": ("Fast semantic search", "music device"),
    "Hybrid": ("Keyword + meaning", "sports shoes"),
    "Query Expansion": ("Adds similar words", "speaker → audio"),
    "Weighted Hybrid": ("Weighted ranking", "better accuracy"),
    "Ensemble": ("Combine all methods", "best results")
}

# ==============================
# CACHE PREPROCESSING (STABLE)
# ==============================
@st.cache(allow_output_mutation=True)
def preprocess_data(products):

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(products)

    # Embeddings (NO progress bar → HF fix)
    embeddings = model.encode(products, batch_size=64, show_progress_bar=False)

    # Normalize for FAISS
    faiss.normalize_L2(embeddings)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    # BM25
    tokenized = [p.split() for p in products]
    bm25 = BM25Okapi(tokenized)

    return tfidf, tfidf_matrix, embeddings, index, bm25


@st.cache(allow_output_mutation=True)
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# ==============================
# FILE LOAD
# ==============================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample dataset")
    df = pd.DataFrame({
        "product_name": [
            "iPhone 14 Pro",
            "Samsung Galaxy S23",
            "Nike Running Shoes",
            "Dell Gaming Laptop",
            "Bluetooth Speaker"
        ],
        "category": ["Mobile", "Mobile", "Footwear", "Laptop", "Electronics"],
        "brand": ["Apple", "Samsung", "Nike", "Dell", "JBL"],
        "description": [
            "Latest smartphone",
            "Android flagship phone",
            "Comfort sports shoes",
            "High performance laptop",
            "Portable music device"
        ]
    })

st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ==============================
# COMBINE TEXT
# ==============================
df["combined"] = (
    df["product_name"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["brand"].astype(str) + " " +
    df["description"].astype(str)
)

products = df["combined"].tolist()

# ==============================
# PREPROCESS (ONLY ONCE)
# ==============================
with st.spinner("Processing data..."):
    tfidf, tfidf_matrix, embeddings, index, bm25 = preprocess_data(products)

# ==============================
# SEARCH FUNCTIONS
# ==============================
def keyword_search(q):
    return [(i, 1) for i, p in enumerate(products) if q.lower() in p.lower()]

def regex_search(q):
    return [(i, 1) for i, p in enumerate(products) if re.search(q, p, re.IGNORECASE)]

def boolean_search(q):
    if "AND" in q:
        terms = q.split("AND")
        return [(i, 1) for i, p in enumerate(products)
                if all(t.strip().lower() in p.lower() for t in terms)]
    elif "OR" in q:
        terms = q.split("OR")
        return [(i, 1) for i, p in enumerate(products)
                if any(t.strip().lower() in p.lower() for t in terms)]
    return []

def fuzzy_search(q):
    scores = [(i, fuzz.ratio(q, p)) for i, p in enumerate(products)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:10]

def ngram_search(q):
    return [(i, 1) for i, p in enumerate(products) if q[:3].lower() in p.lower()]

def prefix_search(q):
    return [(i, 1) for i, p in enumerate(products) if p.lower().startswith(q.lower())]

def suffix_search(q):
    return [(i, 1) for i, p in enumerate(products) if p.lower().endswith(q.lower())]

def tfidf_search(q):
    q_vec = tfidf.transform([q])
    scores = (tfidf_matrix @ q_vec.T).toarray().flatten()
    idx = np.argsort(scores)[::-1][:10]
    return [(i, float(scores[i])) for i in idx]

def bm25_search(q):
    scores = bm25.get_scores(q.split())
    idx = np.argsort(scores)[::-1][:10]
    return [(i, float(scores[i])) for i in idx]

def semantic_search(q):
    q_emb = model.encode([q], show_progress_bar=False)
    faiss.normalize_L2(q_emb)
    scores = np.dot(embeddings, q_emb.T).flatten()
    idx = np.argsort(scores)[::-1][:10]
    return [(i, float(scores[i])) for i in idx]

def faiss_search(q):
    q_emb = model.encode([q], show_progress_bar=False)
    faiss.normalize_L2(q_emb)
    D, I = index.search(np.array(q_emb), 10)
    return [(i, float(D[0][idx])) for idx, i in enumerate(I[0])]

def hybrid_search(q):
    tfidf_res = dict(tfidf_search(q))
    sem_res = dict(semantic_search(q))
    combined = {i: tfidf_res.get(i, 0) + sem_res.get(i, 0) for i in range(len(products))}
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]

def query_expansion_search(q):
    synonyms = get_synonyms(q)
    expanded_query = q + " " + " ".join(synonyms)
    return tfidf_search(expanded_query)

def weighted_hybrid(q):
    tfidf_res = dict(tfidf_search(q))
    sem_res = dict(semantic_search(q))
    bm25_res = dict(bm25_search(q))

    combined = {}
    for i in range(len(products)):
        combined[i] = (
            0.4 * tfidf_res.get(i, 0) +
            0.4 * sem_res.get(i, 0) +
            0.2 * bm25_res.get(i, 0)
        )
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]

def ensemble_search(q):
    results = {}
    for func in [tfidf_search, semantic_search, bm25_search]:
        for i, score in func(q):
            results[i] = results.get(i, 0) + score
    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]

# ==============================
# UI
# ==============================
search_type = st.selectbox("Select Search Type", list(search_info.keys()))

explanation, example = search_info[search_type]

st.markdown(f"""
### 🔍 {search_type} Search
- **Explanation:** {explanation}
- **Example:** `{example}`
""")

query = st.text_input("Enter your search query")

if st.button("Try Example"):
    query = example.split("→")[0].strip()
    st.success(f"Example loaded: {query}")

top_k = st.slider("Top Results", 5, 20, 10)

if st.button("Search"):
    if not query:
        st.warning("Enter query")
    else:
        func_map = {
            "Keyword": keyword_search,
            "Regex": regex_search,
            "Boolean": boolean_search,
            "Fuzzy": fuzzy_search,
            "N-Gram": ngram_search,
            "Prefix": prefix_search,
            "Suffix": suffix_search,
            "TF-IDF": tfidf_search,
            "BM25": bm25_search,
            "Semantic": semantic_search,
            "FAISS": faiss_search,
            "Hybrid": hybrid_search,
            "Query Expansion": query_expansion_search,
            "Weighted Hybrid": weighted_hybrid,
            "Ensemble": ensemble_search
        }

        results = func_map[search_type](query)[:top_k]

        indices = [i for i, _ in results]
        result_df = df.iloc[indices].copy()
        result_df["Score"] = [score for _, score in results]

        st.subheader("🔎 Results")
        st.dataframe(result_df)