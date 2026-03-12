import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


# ==========================================
# 🚀 新增核心模块：多轮对话提问重写 (Query Rewrite)
# ==========================================
def rewrite_question(history_str: str, user_content: str) -> str:
    """
    RAG 历史感知模块：将包含代词的模糊提问重写为独立的精准搜索词。
    """
    # 如果没有历史记录，直接返回原问题，节省 API 算力和时间
    if not history_str or not history_str.strip():
        return user_content

    # 定义专门的重写 Prompt
    rewrite_prompt = PromptTemplate(
        input_variables=["history", "question"],
        template="""你是一个专业的跨语言提问重写专家。请阅读以下历史对话，并将用户的最新提问重写为一个独立、完整、意思明确的句子，以便于在向量数据库中进行精准检索。

        严格遵守以下规则：
        1. 补全原问题中的所有代词（如“它”、“那个”、“这”）。
        2. 补全省略的主语、谓语或宾语。
        3. 保持专业词汇的语种不变（中英混合提问需准确还原术语）。
        4. 如果原问题已经很完整，无需修改，直接原样输出。
        5. **绝对不要回答这个问题！只需输出重写后的句子！**

        【历史对话】
        {history}

        【用户最新提问】
        {question}

        重写后的独立提问："""
    )

    # 构建重写链并确保输出为纯字符串
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    try:
        standalone_question = rewrite_chain.invoke({
            "history": history_str,
            "question": user_content
        })
        return standalone_question.strip()
    except Exception as e:
        print(f"⚠️ 提问重写失败，降级使用原问题: {e}")
        return user_content


# ==========================================
# 3. 增强的检索函数
# ==========================================
def get_relevant_context(query: str) -> str:
    if not retriever: return ""

    # 核心调试：手动执行相似度搜索
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        print(f"⚠️ 警告：针对搜索词 '{query}' 未检索到任何匹配片段！")
        return ""

    print(f"🔎 成功匹配到 {len(docs)} 条参考资料")
    print(f"📄 样例内容: {docs[0].page_content[:50]}...")

    return "\n\n".join([doc.page_content for doc in docs])


# ==========================================
# 4. 生成最终回答
# ==========================================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个具备高级跨语言理解能力的智能问答 Agent。
原则：
1. 【精准检索】：必须优先基于提供的参考知识回答。
2. 【跨语言】：如果参考知识是英文，请用中文准确转述；如果用户用英文提问，请用英文回答。
3. 【诚实】：如果参考知识中没有相关内容，请结合上下文给出合理推断，或直接告知“知识库中未提及”，绝不瞎编。"""),
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