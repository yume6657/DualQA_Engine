import os
import re
import json
import pickle
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
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")

# ── 模型初始化 ──────────────────────────────────────────────────────
llm = ChatOpenAI(model="glm-4-flash", temperature=0.3)
embeddings = OpenAIEmbeddings(model="embedding-3")

# ── 向量库连接 ──────────────────────────────────────────────────────
try:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="bilingual_rag"
    )
    count = vectorstore._collection.count()
    print(f"[INFO] 向量库连接成功！记录数: {count}")
except Exception as e:
    print(f"[ERROR] 向量库连接失败: {e}")
    vectorstore = None

# ── BM25 索引加载 ───────────────────────────────────────────────────
_bm25_data = None
if os.path.exists(BM25_INDEX_PATH):
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
        print(f"[INFO] BM25 索引加载成功！语料片段数: {len(_bm25_data['corpus'])}")
    except Exception as e:
        print(f"[WARN] BM25 索引加载失败: {e}")
else:
    print("[WARN] 未找到 BM25 索引，请先运行 ingest_data.py")


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """与 ingest_data.py 保持一致的双语分词。"""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())


def _reciprocal_rank_fusion(
    vector_docs: list, bm25_texts: list[str], k: int = 60
) -> list[str]:
    """
    RRF 融合排序：将向量检索和 BM25 检索的排名合并为统一分数。
    公式：score(d) = Σ 1/(k + rank_i(d))
    返回按融合分数降序排列的文本列表。
    """
    scores: dict[str, float] = {}

    # 向量检索排名贡献
    for rank, doc in enumerate(vector_docs):
        text = doc.page_content
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)

    # BM25 排名贡献
    for rank, text in enumerate(bm25_texts):
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)

    sorted_texts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in sorted_texts]


# ══════════════════════════════════════════════════════════════════════
# 第一重保障：混合检索（Hybrid Retrieval）
# ══════════════════════════════════════════════════════════════════════

def _vector_search(query: str, k: int = 5) -> list:
    """向量语义检索，返回 Document 列表。"""
    if not vectorstore:
        return []
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        print(f"⚠️ 向量检索失败: {e}")
        return []


def _bm25_search(query: str, k: int = 5) -> list[str]:
    """BM25 关键词检索，返回文本列表。"""
    if not _bm25_data:
        return []
    try:
        tokens = _tokenize(query)
        scores = _bm25_data["bm25"].get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [_bm25_data["corpus"][i] for i in top_indices if scores[i] > 0]
    except Exception as e:
        print(f"⚠️ BM25 检索失败: {e}")
        return []


def hybrid_search(query: str, top_k: int = 5) -> list[str]:
    """
    混合检索：向量检索 + BM25 检索，通过 RRF 融合排序后返回 top_k 结果。
    """
    vector_docs = _vector_search(query, k=top_k)
    bm25_texts = _bm25_search(query, k=top_k)

    if not vector_docs and not bm25_texts:
        return []

    fused = _reciprocal_rank_fusion(vector_docs, bm25_texts)
    return fused[:top_k]


# ══════════════════════════════════════════════════════════════════════
# 第二重保障：相关性过滤（Relevance Gating）
# ══════════════════════════════════════════════════════════════════════

_relevance_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""你是一个严格的相关性评估专家。请判断以下【参考片段】是否包含回答【问题】所需的实质性信息。

【问题】
{question}

【参考片段】
{context}

评分规则（只输出数字，不要任何解释）：
- 2：片段直接包含问题答案或关键事实
- 1：片段与问题话题相关，有一定参考价值
- 0：片段与问题无关或信息量极低

输出（仅一个数字 0/1/2）："""
)

_relevance_chain = _relevance_prompt | llm | StrOutputParser()


def _score_relevance(question: str, context_text: str) -> int:
    """对单个检索片段打相关性分，返回 0/1/2。"""
    try:
        raw = _relevance_chain.invoke({
            "question": question,
            "context": context_text[:300]  # 截断避免超 token
        }).strip()
        score = int(raw[0]) if raw and raw[0] in "012" else 0
        return score
    except Exception:
        return 1  # 评分失败时保守保留


def filter_by_relevance(question: str, candidates: list[str], threshold: int = 1) -> list[str]:
    """
    相关性过滤：对每个候选片段打分，只保留分数 >= threshold 的片段。
    threshold=1 表示至少"话题相关"才保留。
    """
    if not candidates:
        return []

    scored = []
    for text in candidates:
        score = _score_relevance(question, text)
        print(f"  [SCORE] 相关性评分 {score}/2 | 片段: {text[:40]}...")
        if score >= threshold:
            scored.append((score, text))

    # 按分数降序排列
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored]


# ══════════════════════════════════════════════════════════════════════
# 对外接口：双重保障检索
# ══════════════════════════════════════════════════════════════════════

def get_relevant_context(query: str) -> str:
    """
    双重检索精度保障：
      第一重 → 混合检索（向量 + BM25 + RRF 融合）
      第二重 → 相关性过滤（LLM 评分，去除噪声片段）
    返回过滤后的上下文字符串，若无有效结果返回空字符串。
    """
    print(f"\n[SEARCH] 双重检索: {query[:60]}...")

    # 第一重：混合检索
    candidates = hybrid_search(query, top_k=6)
    print(f"  [OK] 混合检索命中 {len(candidates)} 个候选片段")

    if not candidates:
        print("  [WARN] 混合检索无结果，跳过相关性过滤")
        return ""

    # 第二重：相关性过滤
    filtered = filter_by_relevance(query, candidates, threshold=1)
    print(f"  [OK] 相关性过滤后保留 {len(filtered)} 个高质量片段")

    if not filtered:
        print("  [WARN] 所有片段均被过滤，降级为无上下文推理")
        return ""

    # 最终取前 3 个最相关片段
    return "\n\n".join(filtered[:3])


# ═════════════════════════════════════════════════════════════
# 提问重写模块（保持不变）
# ══════════════════════════════════════════════════════════════════════

def rewrite_question(history_str: str, user_content: str) -> str:
    """将含代词的模糊提问重写为独立精准搜索词。"""
    if not history_str or not history_str.strip():
        return user_content

    rewrite_prompt = PromptTemplate(
        input_variables=["history", "question"],
        template="""你是一个专业的跨语言提问重写专家。请阅读以下历史对话，并将用户的最新提问重写为一个独立、完整、意思明确的句子，以便于在向量数据库中进行精准检索。

        严格遵守以下规则：
        1. 补全原问题中的所有代词（如"它"、"那个"、"这"）。
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

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    try:
        return rewrite_chain.invoke({"history": history_str, "question": user_content}).strip()
    except Exception as e:
        print(f"[WARN] 提问重写失败，降级使用原问题: {e}")
        return user_content


# ══════════════════════════════════════════════════════════════════════
# 最终回答生成
# ══════════════════════════════════════════════════════════════════════

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个具备高级跨语言理解能力的智能问答 Agent。
原则：
1. 【精准检索】：必须优先基于提供的参考知识回答。
2. 【跨语言】：如果参考知识是英文，请用中文准确转述；如果用户用英文提问，请用英文回答。
3. 【诚实】：如果参考知识中没有相关内容，请结合上下文给出合理推断，或直接告知"知识库中未提及"，绝不瞎编。"""),
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


# ══════════════════════════════════════════════════════════════════════
# 长期记忆摘要模块
# ══════════════════════════════════════════════════════════════════════

# 摘要配置
SUMMARY_CONFIG = {
    "initial_threshold": 20,      # 首次生成摘要的消息数阈值
    "update_interval": 10,        # 摘要更新间隔（消息数）
    "max_summary_length": 1000,   # 摘要最大字符数
    "compression_enabled": True,  # 是否启用自动压缩
}

_summary_prompt = PromptTemplate(
    input_variables=["messages", "existing_summary"],
    template="""你是一个专业的对话摘要专家。请从以下对话片段中提取关键信息，生成结构化摘要。

【现有摘要】
{existing_summary}

【新对话片段】
{messages}

请提取以下四类信息（JSON 格式输出）：
1. key_facts: 确定的事实、技术栈、配置信息
2. user_preferences: 用户明确的偏好、要求、约束
3. decisions: 已做出的重要决策、技术选型
4. pending_issues: 提及但未解决的问题

输出格式（纯 JSON，不要任何解释）：
{{
  "key_facts": ["事实1", "事实2"],
  "user_preferences": ["偏好1"],
  "decisions": ["决策1"],
  "pending_issues": ["问题1"]
}}"""
)


def generate_summary(messages: list, existing_summary: str = "") -> dict:
    """
    生成或更新对话摘要。

    Args:
        messages: Message 对象列表
        existing_summary: 已有的摘要 JSON 字符串

    Returns:
        dict: 结构化摘要字典
    """
    # 格式化消息为文本
    messages_text = "\n".join([
        f"{'用户' if msg.role == 'user' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 解析现有摘要
    try:
        existing_dict = json.loads(existing_summary) if existing_summary else {
            "key_facts": [],
            "user_preferences": [],
            "decisions": [],
            "pending_issues": []
        }
    except json.JSONDecodeError:
        existing_dict = {
            "key_facts": [],
            "user_preferences": [],
            "decisions": [],
            "pending_issues": []
        }

    # 调用 LLM 生成摘要
    chain = _summary_prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({
            "messages": messages_text,
            "existing_summary": json.dumps(existing_dict, ensure_ascii=False, indent=2)
        })

        # 清理可能的 markdown 代码块标记
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        # 解析 JSON 结果
        new_summary = json.loads(result)

        # 合并摘要（去重）
        for key in existing_dict.keys():
            if key in new_summary:
                existing_dict[key] = list(set(existing_dict[key] + new_summary.get(key, [])))

        return existing_dict

    except Exception as e:
        print(f"[WARN] 摘要生成失败: {e}")
        return existing_dict


def compress_summary(summary_dict: dict, max_length: int = 1000) -> dict:
    """
    当摘要过长时，调用 LLM 进行压缩精简。

    Args:
        summary_dict: 摘要字典
        max_length: 最大字符数

    Returns:
        dict: 压缩后的摘要字典
    """
    current_text = json.dumps(summary_dict, ensure_ascii=False)

    if len(current_text) <= max_length:
        return summary_dict

    compress_prompt = PromptTemplate(
        input_variables=["summary"],
        template="""以下摘要过长，请压缩精简，保留最重要的信息，删除冗余和过时的内容。

【原摘要】
{summary}

输出压缩后的 JSON（格式与原摘要相同）："""
    )

    chain = compress_prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"summary": current_text})

        # 清理可能的 markdown 代码块标记
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        return json.loads(result)
    except Exception as e:
        print(f"[WARN] 摘要压缩失败: {e}")
        return summary_dict


def format_summary_for_prompt(summary_dict: dict) -> str:
    """将摘要字典格式化为适合注入 prompt 的文本。"""
    parts = []

    if summary_dict.get("key_facts"):
        parts.append("关键事实：\n" + "\n".join(f"- {fact}" for fact in summary_dict["key_facts"]))

    if summary_dict.get("user_preferences"):
        parts.append("用户偏好：\n" + "\n".join(f"- {pref}" for pref in summary_dict["user_preferences"]))

    if summary_dict.get("decisions"):
        parts.append("重要决策：\n" + "\n".join(f"- {dec}" for dec in summary_dict["decisions"]))

    if summary_dict.get("pending_issues"):
        parts.append("待解决问题：\n" + "\n".join(f"- {issue}" for issue in summary_dict["pending_issues"]))

    return "\n\n".join(parts)
