import os
import re
import uuid
import shutil
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPU_API_KEY", "")
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_ROOT = os.path.join(BASE_DIR, "media", "temp_kb")

ALLOWED_EXTENSIONS = {".txt", ".pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def _tokenize(text: str) -> list:
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())


def build_knowledge_base(file_path: str, user_id: int) -> dict:
    """
    解析文件，构建向量索引和 BM25 索引，存储到用户专属目录。
    返回 dict: {persist_directory, bm25_index_path, collection_name, chunk_count, kb_dir}
    成功时 chunk_count > 0，失败时抛出异常。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}")

    # 加载文档
    loader = TextLoader(file_path, encoding="utf-8") if ext == ".txt" else PyPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError("文件内容为空，无法构建知识库")

    # 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("文档切分后无有效片段")

    # 准备目录
    kb_id = str(uuid.uuid4())
    kb_dir = os.path.join(KB_ROOT, str(user_id), kb_id)
    persist_dir = os.path.join(kb_dir, "chroma")
    bm25_path = os.path.join(kb_dir, "bm25.pkl")
    source_dir = os.path.join(kb_dir, "source")
    os.makedirs(persist_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)

    # 保存源文件副本
    source_copy = os.path.join(source_dir, os.path.basename(file_path))
    shutil.copy2(file_path, source_copy)

    collection_name = f"kb_{user_id}_{kb_id[:8]}"

    # 向量索引
    embeddings = OpenAIEmbeddings(model="embedding-3")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    count = vectorstore._collection.count()
    if count == 0:
        shutil.rmtree(kb_dir, ignore_errors=True)
        raise ValueError("向量索引构建失败，片段数为 0")

    # BM25 索引
    corpus_texts = [c.page_content for c in chunks]
    bm25 = BM25Okapi([_tokenize(t) for t in corpus_texts])
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus_texts}, f)

    return {
        "kb_dir": kb_dir,
        "persist_directory": persist_dir,
        "bm25_index_path": bm25_path,
        "collection_name": collection_name,
        "chunk_count": count,
        "source_file_path": source_copy,
    }


def delete_knowledge_base_dir(kb_dir: str):
    """删除知识库目录（含向量库和 BM25 索引）。"""
    if kb_dir and os.path.exists(kb_dir):
        shutil.rmtree(kb_dir, ignore_errors=True)


def load_vectorstore(persist_directory: str, collection_name: str):
    """加载已有向量库，失败返回 None。"""
    try:
        embeddings = OpenAIEmbeddings(model="embedding-3")
        vs = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        return vs
    except Exception:
        return None


def load_bm25(bm25_index_path: str):
    """加载已有 BM25 索引，失败返回 None。"""
    try:
        with open(bm25_index_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
