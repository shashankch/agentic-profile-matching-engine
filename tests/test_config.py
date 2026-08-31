import pytest
from unittest.mock import patch, MagicMock
from agentic_profile_matching.config import get_llm_model, SUPPORTED_PROVIDERS


def test_supported_providers_includes_sarvam():
    assert "Sarvam AI" in SUPPORTED_PROVIDERS
    assert "sarvam-105b" in SUPPORTED_PROVIDERS["Sarvam AI"]


@patch("langchain_openai.ChatOpenAI")
def test_get_llm_model_sarvam_ai(mock_chat_openai):
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    model = get_llm_model(
        provider="Sarvam AI",
        model_name="sarvam-105b",
        api_key="test-sarvam-key",
    )

    assert model == mock_instance
    mock_chat_openai.assert_called_once_with(
        model="sarvam-105b",
        api_key="test-sarvam-key",
        base_url="https://api.sarvam.ai/v1",
    )


def test_get_llm_model_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider: InvalidProvider"):
        get_llm_model(
            provider="InvalidProvider",
            model_name="invalid-model",
            api_key="test-key",
        )
