import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# =====================================================================
# 毕设核心亮点：RAG 离线数据处理管道 (Data Pipeline)
# 作用：读取本地文档 -> 文本切分 -> 向量化 (Embedding) -> 存入向量数据库
# =====================================================================

# 1. 加载环境变量并伪装 OpenAI 接口连接智谱
load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

if not api_key:
    raise ValueError("找不到 ZHIPU_API_KEY，请检查 .env 文件！")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

# 2. 初始化嵌入模型 (Embedding Model)
# 我们使用的是智谱的 embedding-3 模型，它专门负责把文字变成高维数学向量
embeddings = OpenAIEmbeddings(model="embedding-3")

# 3. 读取本地文档
# 确保你之前在 knowledge_base 文件夹下建了这个 test_doc.text.txt 文件
file_path = "./knowledge_base/test_doc.text"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"找不到文档: {file_path}，请先创建它！")

print(f"📄 正在读取文档: {file_path} ...")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 4. 文本切分 (Text Splitting)
# 因为大模型的上下文窗口有限，我们需要把长文档切成小块（Chunk）
# chunk_size=100: 每块大约 100 个字符；chunk_overlap=20: 每块之间重叠 20 个字符（防止一句话被从中间硬生生切断）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)
print(f"✂️ 文档被切分成了 {len(chunks)} 个区块。")

# 5. 向量化并存入 ChromaDB 数据库
print("🧠 正在调用智谱 API 计算向量，并存入本地数据库 (请稍候)...")
persist_dir = "./chroma_db"  # 向量数据库在本地的保存路径

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

print(f"✅ 大功告成！向量数据库已成功保存在 {persist_dir} 文件夹中。")