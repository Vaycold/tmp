"""
Final Evaluation Agent (LLM-as-a-Judge)

최종 GAP 결과물을 10개 차원으로 평가합니다.
각 차원별 0~10점, 평균 점수를 최종 agent 점수로 활용합니다.
"""

from models import AgentState, EvaluationResult, DimensionScore
from llm import llm_chat, parse_json


# 10개 평가 차원 정의
EVALUATION_DIMENSIONS = [
    {
        "name": "novelty",
        "label": "참신성",
        "description": "제시된 연구 갭이 기존 연구에서 다루지 않은 새로운 관점을 포함하는가?"
    },
    {
        "name": "specificity",
        "label": "구체성",
        "description": "연구 갭이 모호하지 않고, 구체적인 연구 방향을 제시하는가?"
    },
    {
        "name": "feasibility",
        "label": "실현가능성",
        "description": "제시된 연구 갭을 실제 연구로 수행할 수 있는가? 현실적 제약(데이터, 기술, 비용)을 고려했는가?"
    },
    {
        "name": "groundedness",
        "label": "근거충실성",
        "description": "연구 갭이 실제 논문의 내용과 인용에 기반하여 도출되었는가? 근거 없는 주장이 포함되어 있지 않은가?"
    },
    {
        "name": "impact",
        "label": "학술적 영향력",
        "description": "해당 연구 갭을 해결하면 학술 분야에 의미 있는 기여를 할 수 있는가?"
    },
    {
        "name": "clarity",
        "label": "명확성",
        "description": "연구 갭 문장이 명확하고 이해하기 쉬운가? 전문가가 아닌 사람도 이해할 수 있는가?"
    },
    {
        "name": "relevance",
        "label": "질문 관련성",
        "description": "도출된 연구 갭이 원래 사용자의 연구 질문과 직접적으로 관련이 있는가?"
    },
    {
        "name": "diversity",
        "label": "다양성",
        "description": "여러 연구 갭이 서로 다른 관점과 축을 다루고 있는가? 중복되지 않는가?"
    },
    {
        "name": "evidence_quality",
        "label": "증거 품질",
        "description": "인용된 논문과 증거가 신뢰할 수 있고, 최신 연구를 반영하는가?"
    },
    {
        "name": "actionability",
        "label": "실행가능성",
        "description": "연구 갭을 바탕으로 구체적인 연구 계획이나 실험을 설계할 수 있는가?"
    }
]


def evaluation_node(state: AgentState) -> AgentState:
    """
    LLM-as-a-Judge 기반 최종 결과 평가.

    10개 차원에 대해 0~10점 평가 후 평균 점수 산출.

    Args:
        state: Current agent state

    Returns:
        Updated state with evaluation results
    """
    print(f"\n📊 Final Evaluation Node (LLM-as-a-Judge)")

    if not state["gaps"]:
        print("  ⚠️ No gaps to evaluate")
        state["evaluation"] = EvaluationResult(
            dimension_scores=[],
            average_score=0.0,
            summary="평가할 연구 갭이 없습니다."
        )
        return state

    # GAP 결과 텍스트 구성
    gaps_text = _format_gaps_for_evaluation(state)

    # 각 차원별 평가
    dimension_scores = []

    for dim in EVALUATION_DIMENSIONS:
        try:
            score, reasoning = _evaluate_dimension(
                dim=dim,
                gaps_text=gaps_text,
                user_question=state["user_question"]
            )
            dimension_scores.append(DimensionScore(
                dimension=dim["name"],
                label=dim["label"],
                score=score,
                reasoning=reasoning
            ))
            print(f"  ✓ {dim['label']}: {score}/10")

        except Exception as e:
            state["errors"].append(f"Evaluation error ({dim['name']}): {str(e)}")
            dimension_scores.append(DimensionScore(
                dimension=dim["name"],
                label=dim["label"],
                score=0,
                reasoning=f"평가 실패: {str(e)}"
            ))
            print(f"  ❌ {dim['label']}: 평가 실패")

    # 평균 점수 계산
    valid_scores = [ds.score for ds in dimension_scores if ds.score > 0]
    average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # 종합 평가 요약 생성
    summary = _generate_summary(
        dimension_scores=dimension_scores,
        average_score=average_score,
        user_question=state["user_question"],
        gaps_text=gaps_text
    )

    state["evaluation"] = EvaluationResult(
        dimension_scores=dimension_scores,
        average_score=round(average_score, 2),
        summary=summary
    )

    print(f"\n  ⭐ 종합 점수: {average_score:.2f}/10")
    print(f"  📝 {summary[:100]}...")

    # trace 기록
    state["trace"]["evaluation"] = {
        "scores": {ds.dimension: ds.score for ds in dimension_scores},
        "average": round(average_score, 2)
    }

    return state


def _format_gaps_for_evaluation(state: AgentState) -> str:
    """평가용 GAP 결과 텍스트 구성"""
    lines = []
    lines.append(f"연구 질문: {state['user_question']}")
    lines.append(f"검색 쿼리: {state['refined_query']}")
    lines.append(f"분석 논문 수: {len(state['papers'])}")
    lines.append(f"추출된 한계점 수: {len(state['limitations'])}")
    lines.append("")

    for i, gap in enumerate(state["gaps"], 1):
        lines.append(f"[GAP {i}] 축: {gap.axis}")
        lines.append(f"  갭 문장: {gap.gap_statement}")
        lines.append(f"  근거 논문 수: {len(gap.supporting_papers)}")
        if gap.supporting_quotes:
            for j, quote in enumerate(gap.supporting_quotes[:3], 1):
                lines.append(f"  인용 {j}: {quote[:150]}")
        lines.append("")

    return "\n".join(lines)


def _evaluate_dimension(dim: dict, gaps_text: str, user_question: str) -> tuple:
    """
    개별 차원 평가.

    Returns:
        (score: int, reasoning: str)
    """
    prompt = f"""당신은 학술 연구 품질 평가 전문가입니다.

아래 연구 갭 분석 결과를 '{dim['label']}' 차원에서 평가해주세요.

## 평가 차원
- 차원명: {dim['label']} ({dim['name']})
- 평가 기준: {dim['description']}

## 평가 대상
{gaps_text}

## 원래 연구 질문
{user_question}

## 채점 기준
- 0~2점: 매우 부족 (해당 차원에서 심각한 문제가 있음)
- 3~4점: 부족 (개선이 필요함)
- 5~6점: 보통 (기본적인 수준은 충족)
- 7~8점: 우수 (해당 차원에서 좋은 품질)
- 9~10점: 탁월 (해당 차원에서 매우 뛰어남)

반드시 아래 JSON 형식으로만 응답하세요:
{{"score": 0, "reasoning": "평가 근거를 2~3문장으로 설명"}}"""

    messages = [
        {"role": "system", "content": "당신은 학술 연구 품질 평가 전문가입니다. 반드시 JSON 형식으로만 응답하세요."},
        {"role": "user", "content": prompt}
    ]

    response = llm_chat(messages)
    result = parse_json(response)

    score = int(result.get("score", 0))
    score = max(0, min(10, score))  # 0~10 범위 보정
    reasoning = result.get("reasoning", "평가 근거 없음")

    return score, reasoning


def _generate_summary(dimension_scores: list, average_score: float,
                      user_question: str, gaps_text: str) -> str:
    """종합 평가 요약 생성"""
    scores_text = "\n".join([
        f"- {ds.label}: {ds.score}/10 ({ds.reasoning[:50]}...)"
        for ds in dimension_scores
    ])

    prompt = f"""당신은 학술 연구 품질 평가 전문가입니다.

아래 연구 갭 분석의 차원별 평가 결과를 바탕으로 종합 평가 요약을 작성해주세요.

## 연구 질문
{user_question}

## 차원별 점수
{scores_text}

## 평균 점수: {average_score:.2f}/10

3~5문장으로 종합 평가를 작성하세요. 강점, 약점, 개선 방향을 포함하세요.
반드시 아래 JSON 형식으로만 응답하세요:
{{"summary": "종합 평가 내용"}}"""

    messages = [
        {"role": "system", "content": "당신은 학술 연구 품질 평가 전문가입니다. 반드시 JSON 형식으로만 응답하세요."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = llm_chat(messages)
        result = parse_json(response)
        return result.get("summary", f"평균 점수 {average_score:.2f}/10")
    except Exception:
        return f"평균 점수 {average_score:.2f}/10 (요약 생성 실패)"