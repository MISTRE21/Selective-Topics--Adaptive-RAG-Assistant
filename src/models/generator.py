import os
import re
from typing import List, Dict, Any, Optional

class LLMGenerator:
    """Answer generator with support for:
    1. Google Gemini API (if GEMINI_API_KEY is configured)
    2. Local grounded synthesizer using structured extraction and prompting logic.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                import google.genai as genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Notice: Gemini client init skipped ({e}). Using local grounded synthesizer.")
                self.client = None

    def generate(self, prompt: str, context_chunks: List[str], query: str) -> str:
        """Generate a grounded technical answer given context chunks and the developer query."""
        if self.client:
            try:
                combined_context = "\n\n---\n\n".join(context_chunks)
                system_instruction = (
                    "You are an expert Software Engineering documentation assistant. "
                    "Answer the developer query strictly using the provided context documentation. "
                    "Include concise code snippets where helpful. "
                    "If the answer cannot be found in the context, explicitly state that the documentation does not contain this information."
                )
                full_prompt = f"{system_instruction}\n\nCONTEXT:\n{combined_context}\n\nQUESTION: {query}"
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"API call error ({e}), falling back to local synthesizer.")

        return self._local_grounded_synthesis(query, context_chunks)

    def _local_grounded_synthesis(self, query: str, context_chunks: List[str]) -> str:
        """High-precision local grounded synthesizer that extracts and organizes relevant
        evidence and code snippets from the retrieved context.
        """
        if not context_chunks:
            return "The documentation does not provide information to address this query."

       
        code_blocks: List[str] = []
        explanation_sentences: List[str] = []

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))

        for chunk in context_chunks:
            
            codes = re.findall(r"```[\w]*\n(.*?)```", chunk, re.DOTALL)
            code_blocks.extend([c.strip() for c in codes if c.strip()])

            
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or s_clean.startswith("#") or s_clean.startswith("```"):
                    continue
                s_words = set(re.findall(r"\b\w{3,}\b", s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                if overlap > 0:
                    explanation_sentences.append((overlap, s_clean))

        explanation_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in explanation_sentences[:3]]

        response_parts = []
        if top_sentences:
            response_parts.append(" ".join(top_sentences))
        elif context_chunks:
            
            first_chunk = context_chunks[0]
            clean_text = re.sub(r"#+.*", "", first_chunk).strip()
            response_parts.append(clean_text[:300] + "...")

        if code_blocks:
            best_code = code_blocks[0]
            response_parts.append(f"\n\n```python\n{best_code}\n```")

        return "\n\n".join(response_parts).strip()
