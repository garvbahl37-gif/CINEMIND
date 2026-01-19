import os
import json
import logging
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient

# Configure logging
logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Handles interactions with Hugging Face Inference API for:
    1. Intent Parsing (Filter extraction)
    2. Response Generation (RAG)
    """
    
    def __init__(self):
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            logger.warning("⚠️ HF_TOKEN not found. Chat features will be disabled.")
            self.client = None
        else:
            # Using Qwen2.5-7B-Instruct (High availability on free tier & strong performance)
            self.model_id = "Qwen/Qwen2.5-7B-Instruct"
            self.client = InferenceClient(model=self.model_id, token=self.token)
            logger.info(f"✅ LLM Engine initialized with {self.model_id}")

    def parse_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Extracts search filters from natural language query.
        Returns a dictionary of filters (year_min, year_max, genres, etc.)
        """
        if not self.client:
            return {}

        system_prompt = """
        You are a movie recommendation assistant. Your goal is to extract structured search filters from the user's query.
        Return ONLY a JSON object with the following keys if applicable. Do not explain.
        - "genres": list of strings (e.g. ["Action", "Comedy"])
        - "year_min": int (e.g. 1990)
        - "year_max": int (e.g. 1999)
        
        Example: "Recommend a 90s action movie"
        Output: {"genres": ["Action"], "year_min": 1990, "year_max": 1999}
        """
        
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {user_query}"}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Simple cleanup to find JSON
            import re
            json_match = re.search(r"\{.*\}", content.replace("\n", ""), re.DOTALL)
            if json_match:
                filters = json.loads(json_match.group())
                logger.info(f"Parsed filters: {filters}")
                return filters
            return {}
            
        except Exception as e:
            logger.error(f"Error parsing intent: {e}")
            return {}

    def generate_response(self, user_query: str, candidates: List[Dict], history: List[Dict]) -> str:
        """
        Generates a natural language response based on the search results.
        """
        if not self.client:
            return "I'm sorry, I can't chat right now because my AI brain is missing (HF_TOKEN not set)."

        # Format candidates into a context string
        context_str = ""
        for i, movie in enumerate(candidates[:5]): # Top 5 context
            title = movie.get("title", "Unknown")
            year = movie.get("year", "N/A")
            genes = ", ".join(movie.get("genres", []))
            overview = movie.get("overview", "")[:100] + "..." if movie.get("overview") else "No description."
            context_str += f"{i+1}. {title} ({year}) - {genes}: {overview}\n"

        system_prompt = """
        You are a friendly and knowledgeable movie assistant. 
        Answer the user's request using EXCLUSIVELY the provided movie context.
        
        CRITICAL RULES:
        1. If the User asks for a movie not in the "Context", politely say you couldn't find it in the database.
        2. Do NOT mention any movie that is not listed in the "Context" section below.
        3. Do NOT hallucinate. Ideally, only recommend movies from the list.
        4. Be conversational but honest about what data you have.
        """

        # Build messages for chat API
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add limited history
        for msg in history[-2:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add current context and query
        user_prompt = f"""Context:
{context_str}

User: {user_query}
"""
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error from LLM Provider: {str(e)}"
