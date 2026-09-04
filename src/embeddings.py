"""
Embedding generator wrapper using google-genai.
"""
import time
from typing import List
from google import genai
from google.genai.errors import APIError

def get_embedding(text: str, retries: int = 3, backoff: float = 2.0) -> List[float]:
    """
    Generate vector embeddings for the given text using Google GenAI models.
    Includes rate-limit retry logic.
    """
    client = genai.Client()
    
    for attempt in range(retries):
        try:
            response = client.models.embed_content(
                model='gemini-embedding-2',
                contents=text,
            )
            return response.embeddings[0].values
        except APIError as e:
            # Handle rate limiting (429) and server errors (500, 503)
            err_msg = str(e)
            # If daily quota is exhausted, fail fast without sleeping
            if e.code == 429 and ("free_tier_requests" in err_msg or "daily" in err_msg.lower()):
                raise e
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            raise e
        except Exception as e:
            # Handle generic request failures
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            raise e
            
    raise RuntimeError("Failed to generate embedding after retries")
