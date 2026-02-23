#!/usr/bin/env python3
"""
GAPago - 독립 평가 스크립트 (LLM-as-a-Judge)

파이프라인 결과물(JSON)을 입력받아 10개 차원으로 평가
사용 전, .env 파일 내에 LLM_PROVIDER=openai 지정 후 실행 필요

사용법:
    python evaluate.py outputs/run_YYYYMMDD_HHMMSS.json
    python evaluate.py outputs/run_YYYYMMDD_HHMMSS.json --output outputs/eval_YYYYMMDD.json
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

from config import config
from llm import llm_chat, parse_json


# ═══════════════════════════════════════════════════════════════
# 10개 평가 차원 정의
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# 핵심 함수
# ═══════════════════════════════════════════════════════════════
def load_pipeline_result(filepath: str) -> dict:
    """파이프라인 결과 JSON 로드"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_gaps_for_evaluation(result: dict) -> str:
    """평가용 GAP 텍스트 구성"""
    lines = []
    lines.append(f"연구 질문: {result['question']}")
    lines.append(f"검색 쿼리: {result['query']}")
    lines.append("")

    for i, gap in enumerate(result["gaps"], 1):
        lines.append(f"[GAP {i}] 축: {gap['axis']}")
        lines.append(f"  갭 문장: {gap['gap_statement']}")
        lines.append(f"  근거 논문 수: {len(gap['supporting_papers'])}")
        if gap["supporting_quotes"]:
            for j, quote in enumerate(gap["supporting_quotes"][:3], 1):
                lines.append(f"  인용 {j}: {quote[:200]}")
        lines.append("")

    return "\n".join(lines)


def evaluate_dimension(dim: dict, gaps_text: str, user_question: str) -> dict:
    """개별 차원 평가"""
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

## 응답 규칙
1. 점수를 먼저 결정하세요.
2. reasoning에 반드시 다음 3가지를 포함하세요:
   - [판단 근거] 왜 이 점수를 부여했는지 구체적 이유 (어떤 GAP의 어떤 부분을 보고 판단했는지)
   - [강점] 이 차원에서 잘된 점
   - [개선점] 이 차원에서 부족한 점과 구체적 개선 방향
3. reasoning은 최소 100자 이상 작성하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{"score": 0, "reasoning": "[판단 근거] ... [강점] ... [개선점] ..."}}"""

    messages = [
        {"role": "system", "content": "당신은 학술 연구 품질 평가 전문가입니다. 반드시 JSON 형식으로만 응답하세요."},
        {"role": "user", "content": prompt}
    ]

    response = llm_chat(messages)
    result = parse_json(response)

    score = int(result.get("score", 0))
    score = max(0, min(10, score))
    reasoning = result.get("reasoning", "평가 근거 없음")

    return {
        "dimension": dim["name"],
        "label": dim["label"],
        "score": score,
        "reasoning": reasoning
    }


def generate_summary(dimension_scores: list, average_score: float,
                     user_question: str, gaps_text: str) -> str:
    """종합 평가 요약 생성"""
    scores_text = "\n".join([
        f"- {ds['label']}: {ds['score']}/10\n  근거: {ds['reasoning']}"
        for ds in dimension_scores
    ])

    prompt = f"""당신은 학술 연구 품질 평가 전문가입니다.

아래 연구 갭 분석의 차원별 평가 결과를 바탕으로 종합 평가 요약을 작성해주세요.

## 연구 질문
{user_question}

## 차원별 점수 및 근거
{scores_text}

## 평균 점수: {average_score:.2f}/10

## 응답 규칙
다음 4가지를 반드시 포함하여 5~8문장으로 작성하세요:
1. [전체 요약] 분석의 전반적 수준 한 줄 요약
2. [핵심 강점] 가장 높은 점수를 받은 차원과 그 이유 (점수 명시)
3. [핵심 약점] 가장 낮은 점수를 받은 차원과 그 이유 (점수 명시)
4. [개선 제안] 점수를 높이기 위한 구체적 실행 방안 2~3가지

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


# ═══════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════
def run_evaluation(input_path: str, output_path: str = None):
    """평가 실행"""
    print("\n" + "=" * 60)
    print("📊 GAPago - 최종 평가 (LLM-as-a-Judge)")
    print("=" * 60)

    # 1) 파이프라인 결과 로드
    print(f"\n📂 입력 파일: {input_path}")
    pipeline_result = load_pipeline_result(input_path)

    if not pipeline_result.get("gaps"):
        print("❌ 평가할 연구 갭이 없습니다.")
        return

    print(f"   연구 질문: {pipeline_result['question']}")
    print(f"   GAP 수: {len(pipeline_result['gaps'])}")

    # 2) 평가용 텍스트 구성
    gaps_text = format_gaps_for_evaluation(pipeline_result)

    # 3) 차원별 평가
    print(f"\n🔍 10개 차원 평가 중...\n")
    dimension_scores = []

    for dim in EVALUATION_DIMENSIONS:
        try:
            score_result = evaluate_dimension(
                dim=dim,
                gaps_text=gaps_text,
                user_question=pipeline_result["question"]
            )
            dimension_scores.append(score_result)
            print(f"  ✓ {dim['label']:8s}: {score_result['score']}/10")
        except Exception as e:
            print(f"  ❌ {dim['label']:8s}: 평가 실패 ({e})")
            dimension_scores.append({
                "dimension": dim["name"],
                "label": dim["label"],
                "score": 0,
                "reasoning": f"평가 실패: {str(e)}"
            })

    # 4) 평균 점수
    valid_scores = [ds["score"] for ds in dimension_scores if ds["score"] > 0]
    average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # 5) 종합 평가 요약
    print(f"\n📝 종합 평가 생성 중...")
    summary = generate_summary(
        dimension_scores=dimension_scores,
        average_score=average_score,
        user_question=pipeline_result["question"],
        gaps_text=gaps_text
    )

    # 6) 결과 구성
    eval_result = {
        "input_file": input_path,
        "question": pipeline_result["question"],
        "evaluated_at": datetime.now().isoformat(),
        "llm_provider": config.LLM_PROVIDER,
        "average_score": round(average_score, 2),
        "summary": summary,
        "dimensions": dimension_scores
    }

    # 7) 결과 출력
    print_evaluation(eval_result)

    # 8) 결과 저장
    if not output_path:
        base_dir = os.path.dirname(os.path.abspath(input_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(base_dir, f"eval_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 평가 결과 저장: {output_path}")

    return eval_result


def print_evaluation(eval_result: dict):
    """평가 결과 출력"""
    print("\n" + "=" * 60)
    print("📊 평가 결과")
    print("=" * 60)

    print(f"\n   ⭐ 종합 점수: {eval_result['average_score']}/10\n")

    for dim in eval_result["dimensions"]:
        bar = "█" * dim["score"] + "░" * (10 - dim["score"])
        print(f"   {dim['label']:8s} [{bar}] {dim['score']}/10")
        print(f"            {dim['reasoning']}")
        print()

    print(f"   📝 종합 평가:")
    print(f"   {eval_result['summary']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GAPago 최종 평가 (LLM-as-a-Judge)"
    )
    parser.add_argument(
        "input",
        help="파이프라인 결과 JSON 파일 경로 (예: outputs/run_20250211.json)"
    )
    parser.add_argument(
        "--output", "-o",
        help="평가 결과 저장 경로 (미지정 시 자동 생성)",
        default=None
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    run_evaluation(args.input, args.output)


if __name__ == "__main__":
    main()