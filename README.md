# DualQA Engine

基于 Django + LangChain 的双语 RAG 智能问答引擎，支持中英文混合检索与多轮对话。

## 核心特性

- **双重检索精度保障**
  - 混合检索：向量语义检索 + BM25 关键词检索 + RRF 融合排序
  - 相关性过滤：LLM 评分机制过滤噪声片段

- **智能多轮对话**
  - 提问重写：自动补全代词和省略信息
  - 滑动窗口记忆：保留最近 3 轮对话上下文

- **多格式文档支持**
  - 支持 TXT、PDF 文档解析与向量化
  - 基于 ChromaDB 的持久化向量存储
  - 智能文本分割（chunk_overlap 保证语义完整）

- **用户系统**
  - JWT 身份认证
  - 会话管理与历史记录

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | Django + Django REST Framework |
| AI 引擎 | LangChain + 智谱 AI (GLM-4-Flash) |
| 向量数据库 | ChromaDB |
| Embedding | 智谱 embedding-3 |
| 检索算法 | 向量检索 + BM25 + RRF 融合 |
| 关键词检索 | rank_bm25 (BM25Okapi) |

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
# 创建 .env 文件并添加：
ZHIPU_API_KEY=your_api_key_here
```

### 2. 数据库初始化

```bash
python manage.py makemigrations
python manage.py migrate
```

### 3. 构建知识库

```bash
# 将文档放入 knowledge_base/ 目录
# 支持 .txt 和 .pdf 格式

# 运行索引构建脚本
python ingest_data.py
```

### 4. 启动服务

```bash
python manage.py runserver
```

访问 `http://127.0.0.1:8000` 开始使用。

## API 接口

### 用户注册
```http
POST /api/register/
Content-Type: application/json

{
  "username": "user",
  "password": "pass"
}
```

### 获取 Token
```http
POST /api/token/
Content-Type: application/json

{
  "username": "user",
  "password": "pass"
}
```

### 对话接口
```http
POST /api/chat/
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "你的问题",
  "conversation_id": null  // 新会话传 null，续接会话传会话 ID
}
```

### 会话管理
```http
# 获取会话列表
GET /api/conversations/
Authorization: Bearer <token>

# 获取会话详情
GET /api/conversations/<id>/
Authorization: Bearer <token>

# 删除会话
DELETE /api/conversations/<id>/
Authorization: Bearer <token>
```

## 项目结构

```
DualQA_Engine/
├── chat/                    # 核心应用
│   ├── ai_bot.py           # RAG 检索与生成逻辑
│   ├── models.py           # 数据模型
│   ├── views.py            # API 视图
│   └── serializers.py      # 序列化器
├── knowledge_base/          # 知识库文档目录
├── chroma_db/              # 向量数据库持久化目录
├── templates/              # 前端模板
├── ingest_data.py          # 文档索引构建脚本
├── requirements.txt        # 依赖清单
└── manage.py              # Django 管理脚本
```

## 核心算法

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户提问                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  第一重保障：混合检索 (Hybrid Retrieval)                         │
│  ┌─────────────────────┐     ┌─────────────────────┐           │
│  │    向量语义检索       │     │    BM25 关键词检索   │           │
│  │  (ChromaDB + LLM)   │     │   (rank_bm25)       │           │
│  │  理解同义词、近义词    │     │   精确匹配关键词      │           │
│  └──────────┬──────────┘     └──────────┬──────────┘           │
│             │                           │                       │
│             └───────────┬───────────────┘                       │
│                         ▼                                       │
│             ┌─────────────────────┐                             │
│             │    RRF 融合排序      │                             │
│             │  score = Σ 1/(k+rank)│                             │
│             │  双重命中 → 分数累加  │                             │
│             └──────────┬──────────┘                             │
└────────────────────────┼────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  第二重保障：相关性过滤 (Relevance Gating)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLM 评分链：PromptTemplate → LLM → StrOutputParser      │   │
│  │  对每个片段打分：2=直接相关, 1=话题相关, 0=无关            │   │
│  │  过滤分数 < 1 的噪声片段                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  最终回答生成                                                    │
│  qa_chain = ChatPromptTemplate | llm                            │
│  输入：context + chat_history + user_input                      │
│  输出：基于知识库的精准回答                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 双重检索流程

#### 1. 混合检索 (Hybrid Retrieval)

| 检索方式 | 原理 | 优势 | 劣势 |
|---------|------|------|------|
| 向量检索 | 文本 → Embedding → 余弦相似度 | 理解语义、同义词 | 可能漏掉精确关键词 |
| BM25 检索 | TF-IDF 变体，关键词匹配 | 精确匹配、可解释 | 无法理解语义 |

#### 2. RRF 融合排序

```
公式：score(d) = Σ 1/(k + rank_i(d))

特点：
- 相同文本在字典中去重，分数累加
- 双重命中的片段分数更高，排名更靠前
- 本质是多源投票机制，降低单一检索偏差
```

#### 3. 相关性过滤

```
LLM 评分标准：
- 2 分：片段直接包含问题答案
- 1 分：片段与问题话题相关
- 0 分：片段与问题无关

只保留 ≥ 1 分的片段，取前 3 个传入 LLM
```

### 文本分割策略

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个片段最大 500 字符
    chunk_overlap=50     # 相邻片段重叠 50 字符 (10%)
)
```

**chunk_overlap 的作用**：
- 保证边界信息完整，关键信息不被切断
- 提高检索命中率，完整短语更容易匹配
- 用少量冗余换取信息完整性

### 多轮对话机制

- **滑动窗口**：保留最近 6 条消息（3 轮对话）
- **提问重写**：将含代词的问题重写为独立完整的搜索词
- **上下文注入**：历史对话 + 检索结果 + 当前问题

### LangChain 组件使用

| 组件 | 文件 | 作用 |
|------|------|------|
| `TextLoader` / `PyPDFLoader` | ingest_data.py | 文档加载 |
| `RecursiveCharacterTextSplitter` | ingest_data.py | 文本分割 |
| `OpenAIEmbeddings` | ai_bot.py | 文本向量化 |
| `Chroma` | ai_bot.py | 向量数据库存储与检索 |
| `ChatOpenAI` | ai_bot.py | 大模型调用 |
| `PromptTemplate` | ai_bot.py | 提示词模板 |
| `ChatPromptTemplate` | ai_bot.py | 聊天提示词模板 |
| `StrOutputParser` | ai_bot.py | 输出解析 |
| LCEL (`\|` 操作符) | ai_bot.py | 组件链式组合 |

## 配置说明

### 环境变量

在项目根目录创建 `.env` 文件：

```env
ZHIPU_API_KEY=your_api_key_here
```

### 可调参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | ingest_data.py | 500 | 文本片段大小 |
| `chunk_overlap` | ingest_data.py | 50 | 片段重叠大小 |
| `temperature` | ai_bot.py | 0.3 | LLM 输出随机性 |
| `top_k` | ai_bot.py | 5 | 检索返回数量 |
| `threshold` | ai_bot.py | 1 | 相关性过滤阈值 |

### 模型配置

项目使用智谱 AI 的 API，兼容 OpenAI 接口：

```python
# ai_bot.py
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

llm = ChatOpenAI(model="glm-4-flash", temperature=0.3)
embeddings = OpenAIEmbeddings(model="embedding-3")
```

如需切换到 OpenAI，只需修改 API Key 和 Base URL。

## 常见问题

### Q: 为什么检索结果有重复片段？

A: 由于 `chunk_overlap` 设置，相邻片段会有部分重叠。这是设计预期：
- 保证边界信息完整
- LLM 能自动处理冗余信息
- 提供更完整的上下文

### Q: 如何提高检索精度？

A: 可以调整以下参数：
- 增大 `chunk_overlap` 保证信息完整
- 调高 `threshold` 过滤更多噪声
- 增大 `top_k` 扩大召回范围

### Q: 如何添加新的知识文档？

A: 
1. 将文档放入 `knowledge_base/` 目录
2. 运行 `python ingest_data.py` 重新构建索引

### Q: 支持哪些文档格式？

A: 目前支持 `.txt` 和 `.pdf` 格式。

## 开发记录

- `cb62bca` - 引入双重检索精度保障机制
- `37ddc9b` - 实现提问重写与滑动窗口记忆
- `414af33` - 增加 PDF 文档解析功能
- `005abe1` - 初始化双语 RAG 问答引擎

## 许可证

MIT License
