import os
import shutil
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

if not api_key:
    raise ValueError("找不到 ZHIPU_API_KEY，请检查 .env 文件！")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")


def tokenize_for_bm25(text: str) -> list[str]:
    """
    双语分词：英文按空格切分，中文按字符切分后合并。
    简单高效，无需额外依赖。
    """
    import re
    # 提取所有中文字符序列和英文单词
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
    return tokens


def run_ingest():
    embeddings = OpenAIEmbeddings(model="embedding-3")

    print(f"📂 正在扫描目录: {KNOWLEDGE_DIR}")

    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader
    }

    docs = []
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        print("⚠️ 已自动创建知识库目录，请放入文件后再试。")
        return

    for file in os.listdir(KNOWLEDGE_DIR):
        file_path = os.path.join(KNOWLEDGE_DIR, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            try:
                print(f"📄 正在解析文档: {file} ...")
                loader = loaders[ext](file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ 解析 {file} 出错: {e}")

    if not docs:
        print("❌ 错误：未发现任何有效的 .txt 或 .pdf 文档！")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ 已将文档切分为 {len(chunks)} 个知识片段。")

    # ── 第一重索引：向量库 ──────────────────────────────────────────
    if os.path.exists(PERSIST_DIR):
        print("🧹 发现旧向量索引，正在清理...")
        shutil.rmtree(PERSIST_DIR)

    print("🧠 正在生成向量索引 (请稍候)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="bilingual_rag"
    )
    print(f"✅ 向量索引完成！片段总数: {vectorstore._collection.count()}")

    # ── 第二重索引：BM25 关键词索引 ────────────────────────────────
    print("🔑 正在构建 BM25 关键词索引...")
    corpus_texts = [chunk.page_content for chunk in chunks]
    tokenized_corpus = [tokenize_for_bm25(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus_texts}, f)

    print(f"✅ BM25 索引已保存至: {BM25_INDEX_PATH}")
    print(f"📊 索引库内当前片段总数: {len(corpus_texts)}")


if __name__ == "__main__":
    run_ingest()
