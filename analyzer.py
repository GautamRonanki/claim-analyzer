from openai import (
    OpenAI,
    RateLimitError,
    APIError,
    APIConnectionError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError
)
import json
import time
from datetime import datetime
from config import (
    OPENAI_API_KEY,
    MODEL,
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    DEFAULT_MAX_CLAIMS,
    VALID_UNCERTAINTY_LABELS
)

client = OpenAI(api_key=OPENAI_API_KEY)


def validate_input(text: str, max_claims: int = DEFAULT_MAX_CLAIMS) -> dict:
    """Validate input before processing."""
    
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    
    if not text.strip():
        raise ValueError("Input text cannot be empty or whitespace only")
    
    text_length = len(text)
    if text_length < MIN_TEXT_LENGTH:
        raise ValueError(f"Text too short for analysis (min {MIN_TEXT_LENGTH} chars)")
    if text_length > MAX_TEXT_LENGTH:
        raise ValueError(f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)")
    
    if not isinstance(max_claims, int) or max_claims < 1:
        raise ValueError("max_claims must be a positive integer")
    
    return {
        "text": text.strip(),
        "max_claims": max_claims,
        "text_length": text_length
    }

def call_llm_with_retry(messages, temperature=0.3, max_retries=2):
    """Call OpenAI API with retry logic for transient errors."""
    
    retry_delays = [2, 5] 
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response
            
        except RateLimitError as e:
            if attempt < max_retries:
                delay = retry_delays[attempt]
                print(f"Rate limit hit. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise Exception(f"Rate limit exceeded after {max_retries} retries: {e}")
        
        except (APIError, APIConnectionError, InternalServerError) as e:
            if attempt < max_retries:
                delay = retry_delays[attempt]
                print(f"Server error. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise Exception(f"Server error after {max_retries} retries: {e}")
        
        except (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError) as e:
            raise Exception(f"Invalid request (no retry): {e}")
    
    raise Exception("Unexpected retry loop exit")

def extract_claims(text: str, max_claims: int) -> dict:
    """Use LLM to extract claims with uncertainty labels."""
    
    system_prompt = """You are a precise claim extraction system.

Your job:
1. Read the provided text
2. Extract ALL factual claims or assertions as they appear in the text (not opinions, not questions)
3. DO NOT filter claims based on whether they seem true or reasonable in reality
4. If the text contains contradictory claims, extract BOTH claims separately
5. Your job is to identify what claims are MADE in the text, not whether those claims are correct
6. For each claim, determine uncertainty level based ONLY on the language used in the text:
   - "certain": Definitive statements (is, are, has, will)
   - "likely": Probable statements (appears, seems, suggests)
   - "uncertain": Tentative statements (might, could, may)
   - "speculative": Highly uncertain (possibly, potentially, allegedly)

Output ONLY valid JSON matching this schema:
{
  "claims": [
    {
      "claim": "extracted factual statement",
      "uncertainty": "certain|likely|uncertain|speculative",
      "context": "surrounding sentence for reference"
    }
  ]
}

Rules:
- Extract up to the requested number of claims
- Focus on factual assertions, not opinions
- Use exact uncertainty labels only
- If no claims found, return empty claims array"""

    user_prompt = f"""Extract up to {max_claims} claims from this text:

{text}

Remember: Output valid JSON only."""

    try:
        response = call_llm_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return {
            "claims": result.get("claims", []),
            "tokens_used": response.usage.total_tokens,
            "model": MODEL
        }
        
    except json.JSONDecodeError as e:
        # JSON parsing error
        raise ValueError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        # All other errors
        raise Exception(f"LLM analysis failed: {e}")

def post_process_claims(claims: list) -> list:
    """Validate and clean extracted claims."""
    
    validated_claims = []
    
    for claim in claims:
        # Skip if missing required fields
        if not isinstance(claim, dict):
            continue
        if "claim" not in claim or "uncertainty" not in claim:
            continue
            
        # Enforce valid uncertainty labels
        uncertainty = claim.get("uncertainty", "").lower()
        if uncertainty not in VALID_UNCERTAINTY_LABELS:
            # Default to "uncertain" if invalid
            uncertainty = "uncertain"
        
        validated_claims.append({
            "claim": claim["claim"],
            "uncertainty": uncertainty,
            "context": claim.get("context", "")
        })
    
    return validated_claims


def analyze_text(text: str, max_claims: int = DEFAULT_MAX_CLAIMS) -> dict:
    """Main pipeline: validate → extract → post-process."""
    
    start_time = datetime.now()
    
    try:
        # Step 1: Pre-process (validate input)
        validated_input = validate_input(text, max_claims)
        
        # Step 2: LLM analysis (extract claims)
        extraction_result = extract_claims(
            validated_input["text"],
            validated_input["max_claims"]
        )
        
        # Step 3: Post-process (validate output)
        validated_claims = post_process_claims(extraction_result["claims"])
        
        # Build final output
        result = {
            "claims": validated_claims,
            "metadata": {
                "total_claims_found": len(validated_claims),
                "text_length": validated_input["text_length"],
                "analysis_timestamp": start_time.isoformat(),
                "tokens_used": extraction_result["tokens_used"],
                "model": extraction_result["model"]
            }
        }
        
        return result
        
    except ValueError as e:
        # Input validation errors
        return {
            "error": str(e),
            "error_type": "validation_error"
        }
    except Exception as e:
        # All other errors
        return {
            "error": str(e),
            "error_type": "analysis_error"
        }