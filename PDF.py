from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Any

import streamlit as st
from dotenv import load_dotenv

# Define a Document type alias at module level for type hints
try:
    from langchain_core.documents import Document as LCDocument  # type: ignore
except Exception:
    try:
        from langchain.schema import Document as LCDocument  # type: ignore
    except Exception:
        class LCDocument:  # type: ignore
            pass


def load_env() -> None:
    """Load environment variables from a local .env-like file if present."""
    try:
        load_dotenv()
    except Exception:
        pass


def save_uploaded_pdfs_to_temp(uploaded_files: List[Any]) -> Tuple[str, List[str]]:
    """Save uploaded PDF files to a temporary directory and return its path and file paths."""
    temp_dir = tempfile.mkdtemp(prefix="pdfqa_")
    saved_paths: List[str] = []
    for uf in uploaded_files:
        filename = uf.name
        if not filename.lower().endswith(".pdf"):
            # Only PDFs are allowed
            continue
        file_path = str(Path(temp_dir) / filename)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(file_path)
    return temp_dir, saved_paths


def load_documents(file_paths: List[str], chunk_size: int, chunk_overlap: int) -> List[LCDocument]:
    """Load PDFs into LangChain Documents and split into chunks."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception as exc:
        st.error("Не найдены необходимые пакеты LangChain (loaders/splitter). Установите зависимости.")
        raise exc

    all_docs: List[LCDocument] = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", "? ", "! ", ", ", ",", " "]
    )
    chunked_docs = splitter.split_documents(all_docs)
    return chunked_docs


def make_embeddings(embedding_provider: str, openai_api_key: Optional[str]):
    """Create embeddings object for the chosen provider."""
    if embedding_provider == "OpenAI":
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as exc:
            st.error("Не найдена библиотека 'langchain-openai'. Установите зависимости.")
            raise exc

        effective_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not effective_key:
            st.stop()
        return OpenAIEmbeddings(api_key=effective_key)

    if embedding_provider == "Ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except Exception as exc:
            st.error("Не найдена библиотека 'langchain-community' (OllamaEmbeddings). Установите зависимости.")
            raise exc

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL", "llama3:8b"), base_url=base_url)

    raise ValueError(f"Unknown embedding provider: {embedding_provider}")


def build_vector_store(
    docs: List[LCDocument],
    store_type: str,
    embedding_provider: str,
    openai_api_key: Optional[str],
    persist_dir: Optional[str] = None,
):
    """Build a vector store (Chroma or FAISS) and return its retriever."""
    embeddings = make_embeddings(embedding_provider, openai_api_key)

    if store_type == "Chroma":
        try:
            from langchain_community.vectorstores import Chroma
        except Exception as exc:
            st.error("Не найдена библиотека 'chromadb'/'langchain-community'. Установите зависимости.")
            raise exc

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return vectorstore, retriever

    if store_type == "FAISS":
        try:
            from langchain_community.vectorstores import FAISS
        except Exception as exc:
            st.error("Не найдена библиотека 'faiss-cpu'/'langchain-community'. Установите зависимости.")
            raise exc

        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return vectorstore, retriever

    raise ValueError(f"Unknown store_type: {store_type}")


def make_llm(provider: str, model: str, api_key: Optional[str], temperature: float = 0.0):
    if provider == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            st.error("Не найдена библиотека 'langchain-openai'. Установите зависимости.")
            raise exc

        effective_key = api_key or os.getenv("OPENAI_API_KEY")
        if not effective_key:
            st.stop()
        return ChatOpenAI(model=model, api_key=effective_key, temperature=temperature)

    if provider == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception as exc:
            st.error("Не найдена библиотека 'langchain-ollama'. Установите зависимости.")
            raise exc

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    raise ValueError(f"Unknown provider: {provider}")


def create_qa_chain(retriever, llm):
    try:
        from langchain.chains import RetrievalQA
    except Exception as exc:
        st.error("Не найдена цепочка RetrievalQA из LangChain. Установите зависимости.")
        raise exc

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def main():
    load_env()

    st.set_page_config(page_title="PDF QA", page_icon="📄", layout="centered")
    st.title("📄 AI‑поисковик по PDF")
    st.caption("LangChain + FAISS/Chroma + Streamlit")

    with st.sidebar:
        st.header("Настройки")
        provider = st.selectbox("Провайдер LLM", ["OpenAI", "Ollama"], index=0)
        if provider == "OpenAI":
            model = st.selectbox("Модель (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
            api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        else:
            model = st.text_input("Модель (Ollama)", value=os.getenv("OLLAMA_MODEL", "llama3:8b-instruct"))
            api_key = None

        embedding_provider = st.selectbox("Провайдер эмбеддингов", ["OpenAI", "Ollama"], index=0)
        store_type = st.selectbox("Векторное хранилище", ["Chroma", "FAISS"], index=0)
        chunk_size = st.slider("Размер чанка", 500, 4000, 1500, step=100)
        chunk_overlap = st.slider("Перекрытие", 0, 800, 200, step=50)
        top_k = st.slider("Сколько фрагментов искать (k)", 1, 10, 4)

        persist_chroma = st.checkbox("Сохранять индекс Chroma на диск", value=False)
        persist_dir = st.text_input("Каталог для Chroma (если сохранять)", value=str(Path(tempfile.gettempdir()) / "pdfqa_chroma"))

    st.subheader("Загрузите PDF‑файлы")
    uploaded_files = st.file_uploader("Выберите один или несколько PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("Построить индекс"):
        if not uploaded_files:
            st.warning("Загрузите хотя бы один PDF.")
            st.stop()

        with st.spinner("Сохраняем файлы…"):
            temp_dir, file_paths = save_uploaded_pdfs_to_temp(uploaded_files)

        with st.spinner("Разбиваем и индексируем…"):
            docs = load_documents(file_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            vectorstore, retriever = build_vector_store(
                docs=docs,
                store_type=store_type,
                embedding_provider=embedding_provider,
                openai_api_key=api_key,
                persist_dir=(persist_dir if persist_chroma and store_type == "Chroma" else None),
            )

            # Update retriever k
            retriever.search_kwargs["k"] = top_k

        st.session_state["pdfqa_retriever"] = retriever
        st.session_state["pdfqa_store_type"] = store_type
        st.success("Индекс построен. Можно задавать вопросы.")

    retriever = st.session_state.get("pdfqa_retriever")
    user_question = st.text_input("Вопрос по загруженным PDF")

    if st.button("Искать ответ"):
        if not retriever:
            st.warning("Сначала постройте индекс из PDF.")
            st.stop()

        with st.spinner("Готовим модель…"):
            llm = make_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)

        qa_chain = create_qa_chain(retriever, llm)
        with st.spinner("Ищем ответ…"):
            result = qa_chain({"query": user_question})

        answer = result.get("result", "")
        source_documents = result.get("source_documents", [])

        st.subheader("Ответ")
        st.write(answer)

        if source_documents:
            st.subheader("Источник(и)")
            for i, doc in enumerate(source_documents, start=1):
                metadata = getattr(doc, "metadata", {}) or {}
                source_name = metadata.get("source", "PDF")
                page = metadata.get("page", None)
                page_info = f", страница {page + 1}" if isinstance(page, int) else ""
                with st.expander(f"Фрагмент {i}: {Path(source_name).name}{page_info}"):
                    st.write(doc.page_content)


if __name__ == "__main__":
    main()


