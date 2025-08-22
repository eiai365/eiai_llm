import enum

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser


Chat_Template_Context_Only = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

Chat_Template_Context_Only_Yes_or_No = """Answer the question with Yes or No based only on the following context:
    {context}
    Question: {question}
    """

Chat_Template_Context_Only_Brief_and_Concise = """Answer the question briefly and concisely based only on the following context:
    {context}
    Question: {question}
    """


class PredefinedChatPromptTemplateType(enum.Enum):
    context_only = enum.auto()
    context_only_yes_or_no = enum.auto()
    context_only_brief_and_concise = enum.auto()


class PredefinedChatPromptTemplate:
    @staticmethod
    def get_predefined_chat_template(*, predefined_chat_template_type):
        match predefined_chat_template_type:
            case PredefinedChatPromptTemplateType.context_only:
                return Chat_Template_Context_Only
            case PredefinedChatPromptTemplateType.context_only_yes_or_no:
                return Chat_Template_Context_Only_Yes_or_No
            case PredefinedChatPromptTemplateType.context_only_brief_and_concise:
                return Chat_Template_Context_Only_Brief_and_Concise
            case _:
                return Chat_Template_Context_Only


class PredefinedPromptTemplate:
    @staticmethod
    def get_predefined_prompt_template(*, number):
        template = f"""You are an AI language model assistant. Your task is to generate {number}
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {{question}}"""
        return template


def run(*, llm, question, vector_store, prompt_template_type, prompt_template, question_version_number, chat_prompt_template_type, chat_prompt_template, predefined_chat_template_type, log) -> (bool, str):
    log.info(f"prompt_template_type: {prompt_template_type}")
    match prompt_template_type:
        case "pre_defined":
            prompt_template = PredefinedPromptTemplate.get_predefined_prompt_template(number=question_version_number)
        case "user_defined":
            pass
        case _:
            log.error(f"prompt_template_type {prompt_template_type} is not supported.")
            return False, 'Severe error found.'
    log.info(f"prompt_template: {prompt_template}")

    log.info(f"chat_prompt_template_type: {chat_prompt_template_type}")
    match chat_prompt_template_type:
        case "pre_defined":
            chat_prompt_template = PredefinedChatPromptTemplate.get_predefined_chat_template(predefined_chat_template_type=predefined_chat_template_type)
        case "user_defined":
            pass
        case _:
            log.error(f"chat_prompt_template_type {chat_prompt_template_type} is not supported.")
            return False, 'Severe error found.'

    prompt = PromptTemplate(input_variables=["question"], template=prompt_template, )

    retriever = MultiQueryRetriever.from_llm(vector_store.as_retriever(), llm, prompt=prompt)
    chat_template = chat_prompt_template

    prompt = ChatPromptTemplate.from_template(chat_template)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    response = chain.invoke(question)

    return response
