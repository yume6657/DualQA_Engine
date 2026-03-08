import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

# 获取项目根目录的绝对路径，确保存取路径一致
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")


def run_ingest():
    embeddings = OpenAIEmbeddings(model="embedding-3")

    # 1. 加载所有文本文件
    print(f"📂 扫描目录: {KNOWLEDGE_DIR}")
    loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()

    if not docs:
        print("❌ 错误：未发现任何有效的 .txt 文件！")
        return

    # 2. 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ 已切分出 {len(chunks)} 个知识片段。")

    # 3. 覆盖写入向量库 (清除旧索引以防冲突)
    if os.path.exists(PERSIST_DIR):
        print("🧹 发现旧数据库，正在清理...")
        import shutil
        shutil.rmtree(PERSIST_DIR)

    print("🧠 正在生成向量并构建索引...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="bilingual_rag"  # 显式命名集合
    )
    print(f"✅ 成功！数据库已存至: {PERSIST_DIR}")
    print(f"📊 当前库内片段总数: {vectorstore._collection.count()}")


if __name__ == "__main__":
    run_ingest()