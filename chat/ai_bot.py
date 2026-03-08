import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 新增了 OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma  # 新增了 Chroma

# =====================================================================
# 核心技术点：完整 RAG (检索增强生成) 架构
# =====================================================================

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

if not api_key:
    raise ValueError("找不到 ZHIPU_API_KEY，请检查 .env 文件！")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

# 1. 初始化大模型和向量计算模型
llm = ChatOpenAI(model="glm-4-flash", temperature=0.5)
embeddings = OpenAIEmbeddings(model="embedding-3")

# 2. 连接本地的向量数据库
persist_dir = "./chroma_db"
# as_retriever 设置 k=2，意思是每次根据用户问题，最多搜出最相关的 2 段文档
try:
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    print(f"⚠️ 警告: 向量数据库连接失败，请确认 chroma_db 文件夹是否存在。错误: {e}")
    retriever = None


# 3. 新增核心功能：知识检索函数
def get_relevant_context(query: str) -> str:
    """根据用户的问题，去向量数据库里捞取最相关的私有知识"""
    if not retriever:
        return ""
    try:
        print(f"🔎 正在本地知识库中检索与 '{query}' 相关的内容...")
        docs = retriever.invoke(query)
        # 将搜出来的多段小文本，拼成一个长字符串
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"检索失败: {e}")
        return ""


# 4. Prompt 模板 (保持原有严谨设定)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个具备高级跨语言理解能力的智能问答 Agent。请严格遵循以下原则：
1. 【语言自适应与术语保留】：精准识别用户的提问语言并以此作答。遇到专业术语保留英文原文。
2. 【跨语言知识精准提取】：跨越语言障碍，准确理解下方提供的参考知识并转述。
3. 【多轮语义连贯】：参考历史对话记录。
4. 【严谨求实】：回答必须以提供的参考知识为事实基准。如果知识库中未提及，请直接告知未找到，坚决杜绝大模型幻觉与捏造。"""),
    ("user", """【历史对话记录】\n{chat_history}\n\n【检索到的参考知识】\n{context}\n\n【用户当前问题】\n{user_input}""")
])

qa_chain = prompt_template | llm


# 5. 生成回答函数
def get_bilingual_response(message: str, history: str = "", context: str = "") -> str:
    print(f"\n🧠 正在让 GLM-4 思考问题: '{message}' ...")
    if context:
        print(f"📚 已成功向大模型注入 {len(context)} 个字符的参考知识！")

    response = qa_chain.invoke({
        "user_input": message,
        "chat_history": history,
        "context": context
    })
    return response.content