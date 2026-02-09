"""
LLM provider implementations.
"""

import os
import json
import re
from typing import Optional
from config import config


def mock_llm(messages: list[dict]) -> str:
    """Mock LLM for testing without API."""
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


def openai_llm(messages: list[dict], model: Optional[str] = None) -> str:
    """OpenAI GPT API integration."""
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    model = model or config.OPENAI_MODEL
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ OpenAI API error: {e}")
        raise


def anthropic_llm(messages: list[dict], model: Optional[str] = None) -> str:
    """Anthropic Claude API integration."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    model = model or config.ANTHROPIC_MODEL
    
    # Convert messages format
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
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=system_message if system_message else "You are a helpful research assistant.",
            messages=user_messages
        )
        return response.content[0].text
    except Exception as e:
        print(f"⚠️ Anthropic API error: {e}")
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


# Global cache for Exaone model
_EXAONE_MODEL = None
_EXAONE_TOKENIZER = None


def exaone_llm(messages: list[dict], model: Optional[str] = None) -> str:
    """LG Exaone local model inference."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("PyTorch or Transformers not installed. Run: pip install torch transformers")
    
    if not config.EXAONE_MODEL_PATH:
        raise ValueError("EXAONE_MODEL_PATH not found in environment variables")
    
    global _EXAONE_MODEL, _EXAONE_TOKENIZER
    
    if _EXAONE_MODEL is None:
        print(f"🔄 Loading Exaone model from {config.EXAONE_MODEL_PATH}...")
        
        try:
            _EXAONE_TOKENIZER = AutoTokenizer.from_pretrained(config.EXAONE_MODEL_PATH)
            _EXAONE_MODEL = AutoModelForCausalLM.from_pretrained(
                config.EXAONE_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"✓ Exaone model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load Exaone model: {e}")
            raise
    
    # Format conversation
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
        inputs = _EXAONE_TOKENIZER(conversation, return_tensors="pt").to(_EXAONE_MODEL.device)
        
        with torch.no_grad():
            outputs = _EXAONE_MODEL.generate(
                **inputs,
                max_new_tokens=2000,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=_EXAONE_TOKENIZER.eos_token_id
            )
        
        response = _EXAONE_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        if "[Assistant]" in response:
            response = response.split("[Assistant]")[-1].strip()
        
        return response
    except Exception as e:
        print(f"⚠️ Exaone inference error: {e}")
        raise