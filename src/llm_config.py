"""
Hybrid LLM Configuration
========================

Uses local open-source LLMs (Ollama) for simple tasks
and Gemini for complex medical tasks.

Benefits:
- No API quota limits for query transformation
- No API quota limits for summaries
- Faster response for simple tasks
- Reserve Gemini for complex medical reasoning
"""

import os
from typing import Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseChatModel

# LLM Types
LLMType = Literal["local", "cloud", "auto"]


class HybridLLMConfig:
    """Configuration for hybrid LLM usage"""

    # Local LLM settings (Ollama)
    LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct")  # or "llama3.2:3b"
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Cloud LLM settings (Gemini)
    CLOUD_MODEL = os.getenv("CLOUD_LLM_MODEL", "gemini-2.5-flash")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Task routing
    USE_LOCAL_FOR_SIMPLE_TASKS = os.getenv("USE_LOCAL_FOR_SIMPLE_TASKS", "true").lower() == "true"

    @classmethod
    def get_local_llm(cls, temperature: float = 0.3) -> Ollama:
        """Get local LLM instance (Ollama)"""
        return Ollama(
            model=cls.LOCAL_MODEL,
            base_url=cls.OLLAMA_BASE_URL,
            temperature=temperature
        )

    @classmethod
    def get_cloud_llm(cls, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
        """Get cloud LLM instance (Gemini)"""
        return ChatGoogleGenerativeAI(
            model=cls.CLOUD_MODEL,
            google_api_key=cls.GOOGLE_API_KEY,
            temperature=temperature
        )

    @classmethod
    def get_llm_for_task(cls, task_type: LLMType = "auto", temperature: float = 0.3):
        """
        Get appropriate LLM based on task type.

        Args:
            task_type: "local" (simple tasks), "cloud" (complex), or "auto"
            temperature: LLM temperature

        Returns:
            LLM instance
        """
        if task_type == "local" or (task_type == "auto" and cls.USE_LOCAL_FOR_SIMPLE_TASKS):
            try:
                return cls.get_local_llm(temperature)
            except Exception as e:
                print(f"⚠️  Local LLM not available, falling back to cloud: {e}")
                return cls.get_cloud_llm(temperature)
        else:
            return cls.get_cloud_llm(temperature)

    @classmethod
    def is_local_available(cls) -> bool:
        """Check if local LLM is available"""
        try:
            llm = cls.get_local_llm()
            # Test with a simple query
            llm.invoke("Hello")
            return True
        except Exception:
            return False


def get_simple_task_llm(temperature: float = 0.3):
    """
    Get LLM for simple tasks (query transformation, summaries).
    Uses local LLM if available, falls back to cloud.
    """
    return HybridLLMConfig.get_llm_for_task("auto", temperature)


def get_complex_task_llm(temperature: float = 0.3):
    """
    Get LLM for complex tasks (medical reasoning).
    Always uses cloud LLM (Gemini).
    """
    return HybridLLMConfig.get_cloud_llm(temperature)


# Convenience functions
def get_query_transformation_llm(temperature: float = 0.7):
    """Get LLM for query transformation (local preferred)"""
    return get_simple_task_llm(temperature)


def get_summary_llm(temperature: float = 0.3):
    """Get LLM for summary generation (local preferred)"""
    return get_simple_task_llm(temperature)


def get_medical_reasoning_llm(temperature: float = 0.3):
    """Get LLM for medical reasoning (always cloud)"""
    return get_complex_task_llm(temperature)
