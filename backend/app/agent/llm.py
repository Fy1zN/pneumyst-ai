"""Chat model for the provider named in .env. Only the report and assistant code talk to an LLM."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from backend.app.core.config import Settings

PROVIDERS = ("gemini", "openai")


def chat_model(settings: Settings, model: str) -> BaseChatModel:
    common = {"model": model, "temperature": 0.2, "timeout": settings.llm_timeout_s, "max_retries": 1}
    # Imported here so only the chosen provider's SDK is loaded.
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        thinking = {"thinking_level": settings.llm_thinking_level} if settings.llm_thinking_level else {}
        return ChatGoogleGenerativeAI(google_api_key=settings.llm_api_key, **common, **thinking)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=settings.llm_api_key, **common)


def make_chat_model(settings: Settings) -> Runnable | None:
    if settings.llm_provider not in PROVIDERS:
        raise RuntimeError(f"LLM_PROVIDER must be one of {PROVIDERS}, not {settings.llm_provider!r}")
    if not settings.llm_api_key:
        return None
    if not settings.llm_model:
        raise RuntimeError("LLM_API_KEY is set but LLM_MODEL is not; name an exact model in .env")
    model = chat_model(settings, settings.llm_model)
    if settings.llm_fallback_model:
        # A busy or slow main model gets one try on the fallback before the report uses the template.
        model = model.with_fallbacks([chat_model(settings, settings.llm_fallback_model)])
    return model