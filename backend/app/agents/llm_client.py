"""
llm_client.py — Lightweight, unified LLM client interface for Judge Agents.
Supports OpenAI-compatible APIs (OpenAI, Groq, Ollama, DeepSeek), Gemini REST API,
and local deterministic fallbacks for educational execution without mandatory paid keys.
"""

import json
import logging
import re
from typing import Any, Dict, Optional
import httpx

from app.core import config

logger = logging.getLogger(__name__)


def extract_json_from_response(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON object from model output text.
    Handles markdown code blocks (```json ... ```) or bare JSON objects.
    """
    text = raw_text.strip()
    
    # Try finding markdown code block
    json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_block:
        text = json_block.group(1).strip()
    else:
        # Fallback: find outer curly brackets
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON from LLM output: {raw_text[:200]}...")
        raise ValueError(f"LLM output could not be parsed as JSON: {exc}") from exc


class BaseLLMClient:
    """Base interface for LLM Judge backends."""

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        raw = self.generate(system_prompt, user_prompt, temperature=temperature)
        return extract_json_from_response(raw)


class OpenAICompatibleClient(BaseLLMClient):
    """Client for any OpenAI-compatible Chat Completions endpoint (OpenAI, Groq, Ollama, vLLM)."""

    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.error(f"OpenAI API error {resp.status_code}: {resp.text}")
                raise RuntimeError(f"OpenAI API call failed ({resp.status_code}): {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini REST API."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                logger.error(f"Gemini API error {resp.status_code}: {resp.text}")
                raise RuntimeError(f"Gemini API call failed ({resp.status_code}): {resp.text}")
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]


def get_llm_client() -> Optional[BaseLLMClient]:
    """
    Factory function to get an active LLM client based on configuration.
    Returns None if no external API key is configured (triggering local fallback).
    """
    provider = config.LLM_PROVIDER

    if provider in ("openai", "auto") and config.OPENAI_API_KEY:
        logger.info(f"Using OpenAICompatibleClient with model {config.OPENAI_MODEL}")
        return OpenAICompatibleClient(
            api_key=config.OPENAI_API_KEY,
            api_base=config.OPENAI_API_BASE,
            model=config.OPENAI_MODEL,
        )

    if provider in ("gemini", "auto") and config.GEMINI_API_KEY:
        logger.info(f"Using GeminiClient with model {config.GEMINI_MODEL}")
        return GeminiClient(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
        )

    if provider == "openai" and not config.OPENAI_API_KEY:
        logger.warning("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")

    if provider == "gemini" and not config.GEMINI_API_KEY:
        logger.warning("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set.")

    logger.info("No active external LLM API key detected; using local deterministic semantic judge.")
    return None
