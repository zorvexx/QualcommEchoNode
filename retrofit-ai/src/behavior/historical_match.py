import numpy as np

def match_historical_events(memory_events, current_embedding, top_k=3, min_similarity=70.0):
    """
    Searches historical memory using cosine similarity on embeddings or modality contributions.
    Returns top K matching events.
    """
    if not memory_events or current_embedding is None or len(current_embedding) == 0:
        return []
    
    matches = []
    curr_emb = np.asarray(current_embedding, dtype=np.float32)
    curr_norm = np.linalg.norm(curr_emb) + 1e-9
    
    for event in memory_events:
        hist_emb = event.get('embedding', [])
        if not hist_emb or len(hist_emb) != len(curr_emb):
            continue
            
        h_emb = np.asarray(hist_emb, dtype=np.float32)
        h_norm = np.linalg.norm(h_emb) + 1e-9
        
        sim = float(np.dot(curr_emb, h_emb) / (curr_norm * h_norm))
        sim_pct = round(max(0.0, sim) * 100.0, 1)
        
        if sim_pct >= min_similarity:
            matches.append({
                'event_id': event['event_id'],
                'timestamp': event['timestamp'],
                'similarity': sim_pct,
                'operating_state': event['operating_state'],
                'anomaly_score': event['anomaly_score']
            })
            
    matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    return matches[:top_k]
