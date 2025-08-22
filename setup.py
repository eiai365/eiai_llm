from setuptools import setup, find_packages

setup(
    name="eiai_llm",
    description="EIAI LLM",
    version="0.1.1",
    author="Eiai365 Eiai",
    author_email="eiai365.eiai@gmail.com",
    install_requires=[
        'langchain',
        'langchain-core',
        'langchain-ollama',
        'langchain-community',
        'langchain-chroma',
        'langchain-text-splitters',
        'langchain-ibm',
        'langchain-aws',
        'langchain-google-genai',
        'chromadb',
        'unstructured[all-docs]',
    ],
    packages=find_packages(),
)
