import os
import re
import uuid

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

import eiai_llm.utils as utils

chunk_size = 10000 if os.environ.get("EIAI_LLM_CHUNK_SIZE") is None else int(os.environ.get("EIAI_LLM_CHUNK_SIZE"))
chunk_overlap = 200 if os.environ.get("EIAI_LLM_CHUNK_OVERLAP") is None else int(os.environ.get("EIAI_LLM_CHUNK_OVERLAP"))


def create_vector_db_for_single_file(*, embeddings, persist_directory, file, collection_name, log):
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    collection = vector_db.get()
    existing_ids = collection['ids']
    log.debug(f"exiting ids: {existing_ids}")

    log.debug(file)
    documents = []
    loader = utils.FileLoader.get_file_loader(file=file, log=log)
    if loader is None:
        log.error(f"Cannot get file loader, see previous error messages.")
        return Chroma()

    log.debug(f"file={file}")
    raw_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document = text_splitter.split_documents(raw_document)
    documents.extend(document)

    log.info(f"length: {len(documents)}")

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
    seen_ids = set(existing_ids)
    unique_docs = [doc for doc, i_id in zip(documents, ids) if i_id not in seen_ids and (seen_ids.add(i_id) or True)]
    if len(unique_docs) > 0:
        unique_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in unique_docs]
        Chroma.from_documents(documents=unique_docs, embedding=embeddings, ids=unique_ids, persist_directory=persist_directory, collection_name=collection_name)
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
        collection = vector_db.get()
        log.info(f"now ids: {collection['ids']}")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    return vector_db


def create_vector_db_for_batch_files(*, embeddings, persist_directory, path, collection_name, log):
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    collection = vector_db.get()
    existing_ids = collection['ids']
    log.debug(f"exiting ids: {existing_ids}")
    documents = []
    for file in os.listdir(path):
        log.debug(path)
        log.debug(file)
        if file.endswith(('.pdf', '.txt', '.text', '.md', '.markdown', '.html', '.xlsx')):
            file_p = os.path.join(path, file)
        else:
            log.warning(f"File {os.path.join(path, file)} is not supported.")
            continue
        loader = utils.FileLoader.get_file_loader(file=file_p, log=log)
        if loader is None:
            log.warning(f"Cannot get file loader for {file_p}, see previous error messages.")
            continue

        log.debug(f"file={file_p}")
        raw_document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        document = text_splitter.split_documents(raw_document)
        documents.extend(document)

    log.info(f"length: {len(documents)}")

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
    seen_ids = set(existing_ids)
    unique_docs = [doc for doc, i_id in zip(documents, ids) if i_id not in seen_ids and (seen_ids.add(i_id) or True)]
    if len(unique_docs) > 0:
        unique_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in unique_docs]
        Chroma.from_documents(documents=unique_docs, embedding=embeddings, ids=unique_ids, persist_directory=persist_directory, collection_name=collection_name)
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
        collection = vector_db.get()
        log.info(f"now ids: {collection['ids']}")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    return vector_db


def delete_vector_from_collection_for_batch(*, persist_directory, path, embeddings, collection_name, log):
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    collection = vector_db.get()
    existing_ids = collection['ids']
    log.debug(f"exiting ids: {existing_ids}")
    documents = []
    for file in os.listdir(path):
        log.debug(path)
        log.debug(file)
        if file.endswith(('.pdf', '.txt', '.text', '.md', '.markdown', '.html', '.xlsx')):
            file_p = os.path.join(path, file)
        else:
            log.warning(f"File {os.path.join(path, file)} is not supported.")
            continue
        loader = utils.FileLoader.get_file_loader(file=file_p, log=log)
        if loader is None:
            log.warning(f"Cannot get file loader for {file_p}, see previous error messages.")
            continue
        log.debug(f"file={file_p}")
        raw_document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        document = text_splitter.split_documents(raw_document)
        documents.extend(document)

    log.info(f"lengh: {len(documents)}")

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
    vector_db.delete(ids)


def delete_vector_from_collection_by_document(*, persist_directory, file, embeddings, collection_name, log):
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    collection = vector_db.get()
    existing_ids = collection['ids']
    log.debug(f"exiting ids: {existing_ids}")
    if not re.search(file, ','.join(str(source) for source in collection['metadatas'])):
        log.error(f'File {file} is not be vectored.')
        return None

    loader = utils.FileLoader.get_file_loader(file=file, log=log)
    if loader is None:
        log.error(f"Cannot get file loader, see previous error messages.")
        return None

    raw_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document = text_splitter.split_documents(raw_document)

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in document]
    ids_diff_set = set(ids) - set(existing_ids)
    if not ids_diff_set:
        vector_db.delete(ids)
        return ''.join(ids)
    else:
        log.warning(f"vector with id {''.join(ids)} not existing!")
        return None


def create_vector_store_for_single_file(*, embeddings, file, log):
    log.debug(file)
    documents = []
    loader = utils.FileLoader.get_file_loader(file=file, log=log)
    if loader is None:
        log.error(f"Cannot get file loader, see previous error messages.")
        return Chroma()

    log.debug(f"file={file}")
    raw_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document = text_splitter.split_documents(raw_document)
    documents.extend(document)

    log.info(f"length: {len(documents)}")

    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store


def create_vector_store_for_batch_files(*, embeddings, path, log):
    documents = []
    for file in os.listdir(path):
        log.debug(path)
        log.debug(file)
        if file.endswith(('.pdf', '.txt', '.text', '.md', '.markdown', '.html', '.xlsx')):
            file_p = os.path.join(path, file)
        else:
            log.warning(f"File {os.path.join(path, file)} is not supported.")
            continue
        loader = utils.FileLoader.get_file_loader(file=file_p, log=log)
        if loader is None:
            log.warning(f"Cannot get file loader for {file_p}, see previous error messages.")
            continue

        log.debug(f"file={file_p}")
        raw_document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        document = text_splitter.split_documents(raw_document)
        documents.extend(document)

    log.info(f"length: {len(documents)}")

    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store


def create_vector_store_for_text(*, embeddings, text, log):
    vector_store = Chroma(embedding_function=embeddings)
    collection = vector_store.get()
    existing_ids = collection['ids']
    if len(existing_ids) != 0:
        vector_store.delete(existing_ids)
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document = [Document(page_content=x) for x in text_splitter.split_text(text)]
    documents.extend(document)

    log.info(f"length: {len(documents)}")

    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store
