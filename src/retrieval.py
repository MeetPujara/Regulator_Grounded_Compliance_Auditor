"""
Step 7a: Hybrid Retrieval for Rule Cards.
Plain words: Given a finding, filter by jurisdiction and test_type, then rank 
the rest using meaning (MiniLM + FAISS) and keywords (BM25). Merge and return top 3.
"""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

def load_cards(cards_dir="cards"):
    cards = []
    for p in sorted(Path(cards_dir).glob("*.json")):
        if p.name == "schema.json":
            continue
        cards.append(json.loads(p.read_text(encoding="utf-8")))
    return sorted(cards, key=lambda c: c["id"])

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

def retrieve_cards(finding_jurisdiction, finding_test_type, finding_text, top_k=3):
    cards = load_cards()
    
    # 1. Hard filter
    filtered = [c for c in cards 
                if c["jurisdiction"] == finding_jurisdiction 
                and finding_test_type in c["test_type"]]
    
    if not filtered:
        return []
        
    # 2. Prepare text for ranking (combine clause and obligation)
    docs = [f"{c['clause']}: {c['obligation']}" for c in filtered]
    
    # 3. Keyword ranking (BM25)
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = finding_text.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # 4. Semantic ranking (MiniLM + FAISS)
    model = get_model()
    doc_embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = model.encode([finding_text], convert_to_numpy=True, normalize_embeddings=True)
    
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    _, faiss_scores = index.search(query_embedding, len(filtered))
    faiss_scores = faiss_scores[0] # flatten
    
    # 5. Reciprocal Rank Fusion (RRF)
    bm25_ranks = np.argsort(-bm25_scores)
    faiss_ranks = np.argsort(-faiss_scores)
    
    rank_dict = {c["id"]: {"card": c, "score": 0.0} for c in filtered}
    k = 60 # RRF constant
    
    for rank, idx in enumerate(bm25_ranks):
        card_id = filtered[idx]["id"]
        rank_dict[card_id]["score"] += 1.0 / (k + rank + 1)
        
    for rank, idx in enumerate(faiss_ranks):
        card_id = filtered[idx]["id"]
        rank_dict[card_id]["score"] += 1.0 / (k + rank + 1)
        
    sorted_cards = sorted(rank_dict.values(), key=lambda x: x["score"], reverse=True)
    return [item["card"] for item in sorted_cards[:top_k]]