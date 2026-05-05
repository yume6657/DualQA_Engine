# DualQA Engine

基于 Django + LangChain 的双语 RAG 智能问答引擎，支持中英文混合检索、多轮对话、长期记忆摘要与用户知识库管理。

## 核心特性

- **双重检索精度保障**
  - 混合检索：向量语义检索 + BM25 关键词检索 + RRF 融合排序
  - 相关性过滤：LLM 评分机制（0/1/2 分）过滤噪声片段，只保留 ≥1 分的结果

- **智能多轮对话**
  - 提问重写：自动补全代词和省略信息，生成独立完整的搜索词
  - 滑动窗口记忆：保留最近 6 条消息（3 轮对话）作为短期上下文

- **长期记忆摘要**
  - 对话超过 20 条消息后自动触发摘要生成
  - 每新增 10 条消息自动更新摘要
  - 结构化 JSON 摘要（关键事实、用户偏好、重要决策、待解决问题）
  - 摘要过长时自动压缩，注入 prompt 作为长期记忆

- **用户专属知识库**
  - 每个用户可上传独立的 TXT / PDF 知识库（最大 20MB）
  - 上传后自动构建向量索引（ChromaDB）和 BM25 索引
  - 支持在线预览知识库原文内容
  - 新上传知识库自动替换旧知识库并清理磁盘文件

- **用户系统**
  - JWT 身份认证（access token + refresh token）
  - 会话管理与历史记录（按用户隔离）
  - 个人资料：昵称修改、密码修改、头像上传

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | Django 4.2 + Django REST Framework |
| AI 引擎 | LangChain + 智谱 AI GLM-4-Flash |
| Embedding | 智谱 embedding-3（兼容 OpenAI 接口） |
| 向量数据库 | ChromaDB（用户专属 collection） |
| 关键词检索 | rank-bm25（BM25Okapi） |
| 检索融合 | RRF（Reciprocal Rank Fusion） |
| 认证 | djangorestframework-simplejwt |
| 文档解析 | LangChain TextLoader / PyPDFLoader |
| 前端 | 单页 HTML（原生 JS，无框架依赖） |
| 部署 | Docker + docker-compose |

## 项目结构

```
DualQA_Engine/
├── chat/
│   ├── ai_bot.py               # 核心 AI 逻辑：混合检索、相关性过滤、提问重写、摘要、回答生成
│   ├── knowledge_base_service.py  # 知识库构建：文档解析、向量化、BM25 索引
│   ├── models.py               # 数据模型：Conversation、Message、ConversationSummary、KnowledgeBaseSession、UserProfile
│   ├── views.py                # API 视图：对话、知识库管理、用户资料
│   ├── urls.py                 # URL 路由
│   ├── serializers.py          # 序列化器
│   └── migrations/             # 数据库迁移文件
├── DualQA_Engine/
│   ├── settings.py             # Django 配置
│   └── urls.py                 # 根路由
├── templates/
│   └── chat.html               # 前端单页应用
├── media/
│   ├── temp_kb/                # 用户知识库存储目录（按 user_id/uuid 隔离）
│   └── avatars/                # 用户头像存储目录
├── knowledge_base/             # 全局知识库文档目录（供 ingest_data.py 使用）
├── chroma_db/                  # 全局向量数据库（ingest_data.py 构建）
├── ingest_data.py              # 全局知识库索引构建脚本
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── manage.py
```

## 快速开始

### 方式一：Docker（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 ZHIPU_API_KEY

# 2. 启动服务
docker-compose up --build
```

访问 `http://localhost:8000`

### 方式二：本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
# 创建 .env 文件：
echo "ZHIPU_API_KEY=your_api_key_here" > .env

# 3. 数据库初始化
python manage.py migrate

# 4. 启动服务
python manage.py runserver
```

访问 `http://127.0.0.1:8000`

### （可选）构建全局知识库

```bash
# 将文档放入 knowledge_base/ 目录（支持 .txt 和 .pdf）
python ingest_data.py
```

> 注意：用户也可以在网页端直接上传个人知识库，无需使用此脚本。

## API 接口

所有需要认证的接口须在请求头中携带：
```
Authorization: Bearer <access_token>
```

### 用户注册

```http
POST /api/chat/register/
Content-Type: application/json

{
  "username": "user",
  "password": "pass"
}
```

### 登录（获取 Token）

```http
POST /api/chat/login/
Content-Type: application/json

{
  "username": "user",
  "password": "pass"
}
```

响应：
```json
{
  "access": "<access_token>",
  "refresh": "<refresh_token>"
}
```

### 刷新 Token

```http
POST /api/chat/token/refresh/
Content-Type: application/json

{
  "refresh": "<refresh_token>"
}
```

### 发送消息

```http
POST /api/chat/ask/
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "你的问题",
  "conversation_id": null
}
```

- `conversation_id`：新会话传 `null`，续接已有会话传对应 ID
- 响应包含 `conversation_id`、`user_message`、`ai_message`

### 会话管理

```http
# 获取历史会话列表
GET /api/chat/history/

# 获取某个会话的所有消息
GET /api/chat/history/<id>/

# 删除会话
DELETE /api/chat/history/<id>/delete/
```

### 知识库管理

```http
# 上传知识库（multipart/form-data，字段名 file）
POST /api/chat/knowledge-base/upload/

# 查询当前激活知识库
GET /api/chat/knowledge-base/current/

# 删除当前知识库
DELETE /api/chat/knowledge-base/current/

# 预览知识库原文内容
GET /api/chat/knowledge-base/content/
```

### 用户资料

```http
# 获取个人资料
GET /api/chat/profile/

# 更新昵称 / 修改密码
PUT /api/chat/profile/
Content-Type: application/json

{
  "nickname": "新昵称",
  "old_password": "旧密码",
  "new_password": "新密码"
}

# 上传头像（multipart/form-data，字段名 avatar）
POST /api/chat/profile/avatar/
```

## 核心算法

### 完整问答流程

```
用户提问
   │
   ▼
提问重写（rewrite_question）
   │  将含代词/省略的问题重写为独立完整搜索词
   ▼
混合检索（hybrid_search）
   ├── 向量检索（ChromaDB similarity_search，top-5）
   └── BM25 检索（BM25Okapi，top-5）
         │
         ▼
   RRF 融合排序（score = Σ 1/(60 + rank)）
         │
         ▼
相关性过滤（filter_by_relevance）
   │  LLM 对每个片段打分 0/1/2，保留 ≥1 分，取前 3 个
   ▼
回答生成（get_bilingual_response）
   │  输入：context + 长期摘要 + 最近 6 条消息 + 当前问题
   └── 输出：基于知识库的精准回答
```

### RRF 融合排序

```
score(d) = Σ 1/(k + rank_i(d))，k=60

- 同一文本在向量检索和 BM25 中均命中 → 分数累加，排名更靠前
- 本质是多源投票机制，降低单一检索偏差
```

### 相关性评分标准

| 分数 | 含义 |
|------|------|
| 2 | 片段直接包含问题答案或关键事实 |
| 1 | 片段与问题话题相关，有参考价值 |
| 0 | 片段与问题无关或信息量极低 |

只保留 ≥1 分的片段，最终取前 3 个传入 LLM。

### 长期记忆摘要机制

| 触发条件 | 行为 |
|----------|------|
| 对话消息数首次达到 20 条 | 对前 N-6 条消息生成首次摘要 |
| 摘要后每新增 10 条消息 | 增量更新摘要 |
| 摘要超过 1000 字符 | 自动压缩精简 |

摘要结构（JSON）：
```json
{
  "key_facts": ["确定的事实、技术栈、配置信息"],
  "user_preferences": ["用户明确的偏好、要求"],
  "decisions": ["已做出的重要决策"],
  "pending_issues": ["提及但未解决的问题"]
}
```

### 文本分割策略

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,    # 每个片段最大 500 字符
    chunk_overlap=50   # 相邻片段重叠 50 字符，保证边界信息完整
)
```

### 知识库存储结构

每个用户的知识库独立存储，互不干扰：

```
media/temp_kb/<user_id>/<uuid>/
├── chroma/          # ChromaDB 向量索引
├── bm25.pkl         # BM25 索引（pickle 序列化）
└── source/          # 源文件副本（用于预览）
```

## 配置说明

### 环境变量（.env）

```env
ZHIPU_API_KEY=your_zhipu_api_key_here
```

### 可调参数

| 参数 | 文件 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | knowledge_base_service.py | 500 | 文本片段大小（字符） |
| `chunk_overlap` | knowledge_base_service.py | 50 | 片段重叠大小（字符） |
| `temperature` | ai_bot.py | 0.3 | LLM 输出随机性 |
| `top_k` | ai_bot.py | 5/6 | 检索返回候选数量 |
| `threshold` | ai_bot.py | 1 | 相关性过滤阈值（0/1/2） |
| `initial_threshold` | ai_bot.py | 20 | 首次触发摘要的消息数 |
| `update_interval` | ai_bot.py | 10 | 摘要更新间隔（消息数） |
| `max_summary_length` | ai_bot.py | 1000 | 摘要最大字符数 |
| `MAX_FILE_SIZE` | knowledge_base_service.py | 20MB | 上传文件大小限制 |

### 切换 AI 模型

项目使用智谱 AI 的 OpenAI 兼容接口：

```python
# ai_bot.py / knowledge_base_service.py
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

llm = ChatOpenAI(model="glm-4-flash", temperature=0.3)
embeddings = OpenAIEmbeddings(model="embedding-3")
```

切换到 OpenAI 只需修改 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`，模型名改为 `gpt-4o` / `text-embedding-3-small` 等。

## 数据模型

| 模型 | 说明 |
|------|------|
| `Conversation` | 会话，关联用户，含标题和创建时间 |
| `Message` | 消息，关联会话，含 role（user/assistant）和内容 |
| `ConversationSummary` | 长期记忆摘要，一对一关联会话，JSON 格式存储 |
| `KnowledgeBaseSession` | 用户知识库记录，含文件路径、索引路径、激活状态 |
| `UserProfile` | 用户扩展信息，含昵称和头像 |

## 常见问题

**Q: 没有知识库时能正常对话吗？**

A: 可以。无知识库时跳过检索步骤，直接基于对话历史和 LLM 自身知识回答。

**Q: 如何提高检索精度？**

A: 调整以下参数：
- 增大 `chunk_overlap` 保证边界信息完整
- 调高 `threshold` 为 2，只保留直接相关片段
- 增大 `top_k` 扩大召回范围

**Q: 支持哪些文档格式？**

A: 支持 `.txt`（UTF-8 编码）和 `.pdf` 格式，单文件最大 20MB。

**Q: 多用户之间的知识库会互相影响吗？**

A: 不会。每个用户的知识库存储在独立目录（`media/temp_kb/<user_id>/`），ChromaDB 使用独立 collection，完全隔离。

**Q: 长期摘要会影响回答速度吗？**

A: 摘要生成是异步触发的（在当前请求中同步执行），首次生成摘要时会有轻微延迟。后续对话直接读取已有摘要，无额外开销。

## 许可证

MIT License
