import enum
import json
import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class LLMBackendEnum(enum.Enum):
    ollama = enum.auto()
    aws = enum.auto()
    watsonx = enum.auto()
    google = enum.auto()


class LLMClientFactory:
    @staticmethod
    def create_llm_client(**kwargs):
        match LLMBackendEnum[os.environ["EIAI_LLM_BACKEND"]]:
            case LLMBackendEnum.ollama:
                return LLMClientFactory._create_llm_chat_ollama(), LLMClientFactory._create_llm_embeddings_ollama()
            case LLMBackendEnum.aws:
                return LLMClientFactory._create_llm_chat_aws(), LLMClientFactory._create_llm_embeddings_aws()
            case LLMBackendEnum.watsonx:
                return LLMClientFactory._create_llm_chat_watsonx(**kwargs), LLMClientFactory._create_llm_embeddings_watsonx(**kwargs)
            case LLMBackendEnum.google:
                return LLMClientFactory._create_llm_chat_google(), LLMClientFactory._create_llm_embeddings_google()
            case _:
                raise KeyError(
                    f'Requested LLM backend is not supported: {os.environ["EIAI_LLM_BACKEND"]}.'
                    f"Supported LLM backends are: {[item.name for item in LLMBackendEnum]}"
                )

    @staticmethod
    def _create_llm_chat_watsonx(**kwargs):
        defaults = {
            "url": os.environ.get("EIAI_LLM_WATSONX_URL"),
            "apikey": os.environ.get("EIAI_LLM_WATSONX_APIKEY"),
            "project_id": os.environ.get("EIAI_LLM_WATSONX_PROJECT"),
            "model_id": os.environ.get("EIAI_LLM_FOUNDATION_MODEL"),
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 1,
        }
        return ChatWatsonx(
            **(defaults | kwargs),
        )

    @staticmethod
    def _create_llm_embeddings_watsonx(**kwargs):
        defaults = {
            "url": os.environ.get("EIAI_LLM_WATSONX_URL"),
            "apikey": os.environ.get("EIAI_LLM_WATSONX_APIKEY"),
            "project_id": os.environ.get("EIAI_LLM_WATSONX_PROJECT"),
            "model_id": os.environ.get("EIAI_LLM_EMBEDDING_MODEL"),
        }
        return WatsonxEmbeddings(
            **(defaults | kwargs),
        )

    @staticmethod
    def _create_llm_chat_ollama():
        defaults = json.loads(f'{{"base_url": "localhost:11434", "model": "{os.environ["EIAI_LLM_FOUNDATION_MODEL"]}"}}')
        return ChatOllama(
            **(defaults | json.loads(os.environ.get("EIAI_LLM_OLLAMA_CHAT_PARAMS", "{}"))),
        )

    @staticmethod
    def _create_llm_embeddings_ollama():
        defaults = json.loads(f'{{"model": "{os.environ["EIAI_LLM_EMBEDDING_MODEL"]}"}}')
        return OllamaEmbeddings(**(defaults | json.loads(os.environ.get("EIAI_LLM_OLLAMA_EMBEDDINGS_PARAMS", "{}"))), )

    @staticmethod
    def _create_llm_chat_aws():
        return ChatBedrock(model=os.environ.get("EIAI_LLM_FOUNDATION_MODEL"), region=os.environ.get("EIAI_LLM_AWS_REGION"))

    @staticmethod
    def _create_llm_embeddings_aws():
        return BedrockEmbeddings(model_id=os.environ.get("EIAI_LLM_EMBEDDING_MODEL"), region_name=os.environ.get("EIAI_LLM_AWS_REGION"))

    @staticmethod
    def _create_llm_chat_google():
        return ChatGoogleGenerativeAI(model=os.environ.get("EIAI_LLM_FOUNDATION_MODEL"))

    @staticmethod
    def _create_llm_embeddings_google():
        return GoogleGenerativeAIEmbeddings(model=os.environ.get("EIAI_LLM_EMBEDDING_MODEL"))
