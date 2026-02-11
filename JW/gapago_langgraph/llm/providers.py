"""
LLM implementations
- 모든 함수는 동일한 Contract를 가짐
    - Input : messages:(각 agent에서 정의된 프롬프트[{"role": "...", "content": "..."}])
    - Output: 모델이 생성한 텍스트 응답(str)
- 구현 목록:
    - openai_llm          : GPT API (Azure OpenAI)
    - bedrock_claude_llm  : Claude API (AWS Bedrock Claude)
    - gemini_llm          : Gemini API
    - exaone_llm          : LG Exaone 로컬 추론 (Transformers)
    - mock_llm            : API 없이 파이프라인 E2E 테스트용 더미 응답 생성
"""

import os
import json
import re
from typing import Optional
from config import config
from openai import AzureOpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def openai_llm(messages: list[dict]) -> str:
    """Azure OpenAI API integration."""

    ## API 및 Endporint 누락 시 조기 에러 
    if not config.AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
    if not config.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")

    ## 클라이언트 호출 및 답변 생성
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT
    )
    deployment = model or config.AZURE_OPENAI_DEPLOYMENT
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=2000
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"⚠️ Azure OpenAI API error: {e}")
        raise

class ExaoneClient:
    """Exaone 모델 관리 (로드 1회, 이후 캐싱)"""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """모델 최초 로드"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = config.EXAONE_MODEL_PATH
        print(f"🔄 Loading Exaone model from {model_path}...")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("✓ Exaone model loaded successfully")
        except Exception as e:
            self._model = None
            self._tokenizer = None
            print(f"❌ [모델 로드 실패] {e}")
            raise
    
    def unload(self):
        """모델 메모리 해제"""
        import torch
        
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            print("✓ Exaone model unloaded")
    
    def chat(self, messages: list[dict]) -> str:
        """LLM 호출"""
        import torch
        
        if not config.EXAONE_MODEL_PATH:
            raise ValueError("[설정 오류] EXAONE_MODEL_PATH가 .env에 설정되지 않았습니다.")
        
        # 최초 호출 시 모델 로드
        if self._model is None:
            self._load_model()
        
        # 대화 포맷 구성
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                conversation += f"[System]\n{content}\n\n"
            elif role == "user":
                conversation += f"[User]\n{content}\n\n"
            elif role == "assistant":
                conversation += f"[Assistant]\n{content}\n\n"
        conversation += "[Assistant]\n"
        
        try:
            inputs = self._tokenizer(conversation, return_tensors="pt").to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "[Assistant]" in response:
                response = response.split("[Assistant]")[-1].strip()
            
            return response
        except Exception as e:
            print(f"❌ [추론 오류] {e}")
            raise


# 싱글톤 인스턴스
_exaone_client = ExaoneClient()


def exaone_llm(messages: list[dict]) -> str:
    """Exaone LLM 호출 (기존 인터페이스 유지)"""
    return _exaone_client.chat(messages)


def bedrock_claude_llm(messages: list[dict], model: Optional[str] = None) -> str:
    """
    AWS Bedrock Claude API integration.
    
    Based on: https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
    
    Requires:
        pip install boto3
        AWS credentials configured via:
          - aws configure (recommended)
          - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
          - IAM role (if running on EC2/Lambda)
    
    Environment Variables (optional):
        AWS_REGION (default: us-east-1)
        BEDROCK_CLAUDE_MODEL (default: us.anthropic.claude-3-5-sonnet-20241022-v2:0)
    
    Available Models (2024):
        - us.anthropic.claude-3-5-sonnet-20241022-v2:0 (latest, recommended)
        - us.anthropic.claude-3-5-sonnet-20240620-v1:0
        - anthropic.claude-3-5-sonnet-20240620-v1:0
        - anthropic.claude-3-sonnet-20240229-v1:0
        - anthropic.claude-3-haiku-20240307-v1:0
        - anthropic.claude-3-opus-20240229-v1:0
    """
    try:
        import boto3
        import json as json_lib
    except ImportError:
        raise ImportError("Boto3 not installed. Run: pip install boto3")
    
    # Get region from environment or use default
    region = os.getenv("AWS_REGION", "us-east-1")
    
    # Initialize Bedrock client
    try:
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )
    except Exception as e:
        raise ValueError(
            f"Failed to initialize AWS Bedrock client: {e}\n"
            "Please configure AWS credentials:\n"
            "  1. Run 'aws configure' (recommended)\n"
            "  2. Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )
    
    # Model ID - 올바른 형식 사용
    model_id = model or os.getenv(
        "BEDROCK_CLAUDE_MODEL", 
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # 🔥 수정: 최신 모델 ID
    )
    
    # Convert messages format for Bedrock Claude
    system_message = None
    user_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            user_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Construct Bedrock request body
    # Reference: https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": user_messages,
        "temperature": 0.7
    }
    
    if system_message:
        body["system"] = system_message
    
    try:
        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json_lib.dumps(body)
        )
        
        # Parse response
        response_body = json_lib.loads(response['body'].read())
        
        # Extract text from response
        # Bedrock Claude response format:
        # {
        #   "id": "msg_xxx",
        #   "type": "message",
        #   "role": "assistant",
        #   "content": [{"type": "text", "text": "..."}],
        #   "model": "claude-3-5-sonnet-20241022",
        #   "stop_reason": "end_turn",
        #   "usage": {...}
        # }
        if 'content' in response_body and len(response_body['content']) > 0:
            # content is a list of content blocks
            text_blocks = [
                block['text'] for block in response_body['content']
                if block.get('type') == 'text'
            ]
            return ''.join(text_blocks)
        else:
            raise ValueError(f"Unexpected response format from Bedrock: {response_body}")
        
    except Exception as e:
        print(f"⚠️ AWS Bedrock Claude API error: {e}")
        
        # More detailed error message
        if "ValidationException" in str(e):
            print(f"\n💡 Tip: Make sure the model '{model_id}' is available in region '{region}'")
            print("   Available models vary by region. Check:")
            print("   https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html")
        elif "AccessDeniedException" in str(e):
            print(f"\n💡 Tip: Request model access in AWS Console:")
            print("   1. Go to AWS Bedrock Console")
            print("   2. Navigate to 'Model access'")
            print("   3. Request access to Anthropic Claude models")
        
        raise


def gemini_llm(messages: list[dict], model: Optional[str] = None) -> str:
    """Google Gemini API integration."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Google GenerativeAI package not installed. Run: pip install google-generativeai")
    
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = model or config.GEMINI_MODEL
    gemini_model = genai.GenerativeModel(model)
    
    # Convert messages
    conversation_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            conversation_parts.append(f"System: {content}\n\n")
        elif role == "user":
            conversation_parts.append(content)
    
    prompt = "".join(conversation_parts)
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2000,
            )
        )
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
        raise



def mock_llm(messages: list[dict]) -> str:
    """
    - Mock LLM for testing without API: 미리 설정된 가짜 응답을 반환하도록 시뮬레이션
    """
    content = messages[-1]["content"].lower()
    
    # Query refinement
    if "refine" in content and "search query" in content:
        question_match = re.search(r"research question:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        question = question_match.group(1).strip() if question_match else "research question"
        
        words = re.findall(r'\b\w{4,}\b', question.lower())
        keywords = list(set(words[:5]))
        
        return json.dumps({
            "refined_query": " ".join(keywords[:3]),
            "keywords": keywords[:5],
            "negative_keywords": ["review", "survey"]
        })
    
    # Limitation extraction
    elif "limitation" in content or "future work" in content:
        abstract_match = re.search(r"abstract:\s*(.+?)(?:\n\n|$)", content, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            sentences = re.split(r'[.!?]+', abstract)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            limitations = []
            limit_keywords = ["however", "limitation", "future", "challenge", "difficult", "cannot", "limited"]
            
            for sent in sentences:
                if any(kw in sent.lower() for kw in limit_keywords):
                    limitations.append({
                        "claim": f"Limitation identified: {sent[:80]}...",
                        "evidence_quote": sent[:150]
                    })
                    if len(limitations) >= 2:
                        break
            
            if not limitations and sentences:
                limitations.append({
                    "claim": f"Potential limitation: {sentences[-1][:80]}...",
                    "evidence_quote": sentences[-1][:150]
                })
            
            return json.dumps({"limitations": limitations})
    
    # Axis classification
    elif "classify" in content and "axis" in content:
        claim_lower = content.lower()
        
        if "data" in claim_lower or "dataset" in claim_lower:
            axis = "data_dependency"
        elif "robust" in claim_lower or "noise" in claim_lower:
            axis = "robustness"
        elif "scale" in claim_lower or "large" in claim_lower:
            axis = "scalability"
        elif "general" in claim_lower or "transfer" in claim_lower:
            axis = "generalization"
        elif "evaluat" in claim_lower or "metric" in claim_lower:
            axis = "evaluation_gap"
        elif "practic" in claim_lower or "deploy" in claim_lower:
            axis = "practicality"
        elif "interpret" in claim_lower or "explain" in claim_lower:
            axis = "interpretability"
        else:
            axis = "methodology_gap"
        
        return json.dumps({"axis": axis})
    
    # GAP generation
    elif "research gap" in content or "gap statement" in content:
        axis_match = re.search(r"'(\w+)'", content)
        axis = axis_match.group(1) if axis_match else "methodology_gap"
        
        gap_templates = {
            "data_dependency": "Current approaches require large-scale datasets which limits applicability to low-resource scenarios.",
            "robustness": "Existing models lack robustness to domain shifts and adversarial perturbations.",
            "scalability": "Computational requirements prevent scaling to larger problem instances.",
            "generalization": "Models demonstrate limited generalization across different domains and tasks.",
            "evaluation_gap": "Evaluation metrics do not adequately capture real-world performance.",
            "practicality": "Deployment barriers exist due to resource constraints and integration challenges.",
            "interpretability": "Lack of interpretability hinders trust and adoption in critical applications.",
            "methodology_gap": "Methodological limitations restrict the scope of applicable scenarios."
        }
        
        return json.dumps({
            "gap_statement": gap_templates.get(axis, "Research gap identified in this area.")
        })
    
    return json.dumps({"result": "mock response"})

