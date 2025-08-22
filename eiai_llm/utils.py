import os

from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader, UnstructuredExcelLoader


class FileLoader:
    @staticmethod
    def get_file_loader(*, file, log):
        match os.path.splitext(file)[1]:
            case ".pdf":
                loader = UnstructuredPDFLoader(file)
            case ".txt" | ".text":
                loader = TextLoader(file)
            case ".md" | ".markdown":
                loader = UnstructuredMarkdownLoader(file)
            case ".html" | ".markdown":
                loader = UnstructuredHTMLLoader(file)
            case ".xlsx" | ".markdown":
                loader = UnstructuredExcelLoader(file)
            case _:
                log.error(f"file {file} is not supported.")
                loader = None
        return loader
