# eiai_llm

Backend LLM can be ollama, google, aws or IBM watsonx

Environment Variable:

EIAI_LLM_BACKEND=(ollama/google/aws/watsonx)

watsonx:
EIAI_LLM_WATSONX_URL=
EIAI_LLM_WATSONX_APIKEY=
EIAI_LLM_WATSONX_PROJECT=

aws:
AWS_BEARER_TOKEN_BEDROCK=
EIAI_LLM_AWS_REGION=

google:
GOOGLE_API_KEY=

2 generic Environment Variables:
EIAI_LLM_FOUNDATION_MODEL=
EIAI_LLM_EMBEDDING_MODEL=

Samples:
watsonx:
EIAI_LLM_FOUNDATION_MODEL=meta-llama/llama-3-2-3b-instruct
EIAI_LLM_EMBEDDING_MODEL=ibm/granite-embedding-278m-multilingual

ollama:
EIAI_LLM_FOUNDATION_MODEL=llama3.2:3b
EIAI_LLM_EMBEDDING_MODEL=granite-embedding:278m

aws:
EIAI_LLM_FOUNDATION_MODEL=amazon.titan-text-express-v1
EIAI_LLM_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

google:
EIAI_LLM_FOUNDATION_MODEL=gemini-2.0-flash
EIAI_LLM_EMBEDDING_MODEL=models/gemini-embedding-001
