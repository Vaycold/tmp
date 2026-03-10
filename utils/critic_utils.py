# utils/critic_utils.py

def _compute_critic_scores(
    refined_query: str,
    papers: list,
    limitations: list
) -> dict:
    """
    critic_score_node 전용 내부 함수.
    query_specificity, paper_relevance, groundedness 점수 계산.

    Returns:
        {
            "query_specificity": float,
            "paper_relevance": float,
            "groundedness": float
        }
    """

    # 1. query_specificity: 불용어 제외 content word 비율
    stopwords = {"the", "a", "is", "of", "in", "and", "to", "for"}
    words = refined_query.split()
    content_words = [w for w in words if w.lower() not in stopwords]
    query_spec = min(len(content_words) / max(len(words), 1), 1.0)

    # 2. paper_relevance: 최고 BM25 기준 상대적 정규화
    if papers:
        scores = [p.score_bm25 for p in papers]
        max_score = max(scores) if scores else 1.0
        paper_rel = min(sum(scores) / (len(scores) * max_score), 1.0)
    else:
        paper_rel = 0.0

    # 3. groundedness: evidence_quote 존재 + 5단어 이상 품질 검증
    def _is_valid_evidence(lim) -> bool:
        if not lim.evidence_quote:
            return False
        return len(lim.evidence_quote.split()) >= 5

    if limitations:
        grounded = sum(
            1 for lim in limitations if _is_valid_evidence(lim)
        ) / len(limitations)
    else:
        grounded = 0.0

    return {
        "query_specificity": round(query_spec, 4),
        "paper_relevance":   round(paper_rel, 4),
        "groundedness":      round(grounded, 4),
    }