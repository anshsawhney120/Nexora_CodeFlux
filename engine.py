import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from parsing import preprocess_for_bm25, preprocess_for_semantics

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_ranks(scores):
    """Converts raw scores to 1-based ranks (1 = highest score)."""
    scores = np.array(scores)
    temp = np.argsort(-scores)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(scores))
    return ranks + 1

def calculate_ovr(bm25_scores, semantic_scores, k=60):
    """Applies Reciprocal Rank Fusion and Min-Max scaling for the final OVR."""
    bm25_ranks = get_ranks(bm25_scores)
    semantic_ranks = get_ranks(semantic_scores)
    
    rrf_scores = (1 / (k + bm25_ranks)) + (1 / (k + semantic_ranks))
    
    rrf_min = np.min(rrf_scores)
    rrf_max = np.max(rrf_scores)
    
    if rrf_max == rrf_min:
        return np.full_like(rrf_scores, 85.0)
        
    ovr_scores = ((rrf_scores - rrf_min) / (rrf_max - rrf_min)) * 100
    return np.round(ovr_scores, 0)

def rank_candidates(jd_text, candidates_data):
    """Calculates independent scores, applies RRF, and returns the sorted candidates."""
    jd_bm25 = preprocess_for_bm25(jd_text)
    jd_semantic = preprocess_for_semantics(jd_text)
    jd_vector = embedding_model.encode([jd_semantic])
    
    corpus_bm25 = [preprocess_for_bm25(c['raw_text']) for c in candidates_data]
    semantic_texts = [preprocess_for_semantics(c['raw_text']) for c in candidates_data]
    
    bm25_engine = BM25Okapi(corpus_bm25)
    raw_bm25_scores = bm25_engine.get_scores(jd_bm25)
    
    resume_vectors = embedding_model.encode(semantic_texts)
    raw_semantic_scores = cosine_similarity(jd_vector, resume_vectors)[0]
    
    final_ovrs = calculate_ovr(raw_bm25_scores, raw_semantic_scores)
    
    for idx, candidate in enumerate(candidates_data):
        candidate['ovr'] = final_ovrs[idx]
        candidate['bm25_rank'] = get_ranks(raw_bm25_scores)[idx]
        candidate['semantic_rank'] = get_ranks(raw_semantic_scores)[idx]
        
    candidates_data.sort(key=lambda x: x['ovr'], reverse=True)
    return candidates_data