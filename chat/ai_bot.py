import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# 1. 初始化模型
llm = ChatOpenAI(model="glm-4-flash", temperature=0.3)
embeddings = OpenAIEmbeddings(model="embedding-3")

# 2. 连接向量库 (增加存量检查)
try:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="bilingual_rag"
    )
    count = vectorstore._collection.count()
    print(f"🔗 向量库连接成功！绝对路径: {PERSIST_DIR}")
    print(f"📈 数据库在线记录数: {count}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"❌ 数据库连接失败: {e}")
    retriever = None


# 3. 增强的检索函数
def get_relevant_context(query: str) -> str:
    if not retriever: return ""

    # 核心调试：手动执行相似度搜索
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        print(f"⚠️ 警告：针对问题 '{query}' 未检索到任何匹配片段！")
        return ""

    print(f"🔎 成功匹配到 {len(docs)} 条参考资料")
    # 打印第一条匹配内容的前 50 字供验证
    print(f"📄 样例内容: {docs[0].page_content[:50]}...")

    return "\n\n".join([doc.page_content for doc in docs])


# 4. 优化后的提示词
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个具备高级跨语言理解能力的智能问答 Agent。
原则：
1. 【精准检索】：必须优先基于提供的参考知识回答。
2. 【跨语言】：如果参考知识是英文，请用中文准确转述。
3. 【诚实】：如果参考知识中没有相关内容，请直接告知“知识库中未提及”，不要瞎编。"""),
    ("user", """【参考知识】\n{context}\n\n【对话历史】\n{chat_history}\n\n【当前问题】\n{user_input}""")
])

qa_chain = prompt_template | llm


def get_bilingual_response(message: str, history: str = "", context: str = "") -> str:
    response = qa_chain.invoke({
        "user_input": message,
        "chat_history": history,
        "context": context
    })
    return response.content