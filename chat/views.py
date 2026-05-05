from rest_framework.decorators import api_view, parser_classes, authentication_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.authentication import JWTAuthentication
import os
import tempfile
from .models import Conversation, Message, ConversationSummary, KnowledgeBaseSession, UserProfile
from .knowledge_base_service import build_knowledge_base, delete_knowledge_base_dir, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from .serializers import MessageSerializer, UserSerializer
import json

# 引入我们在 ai_bot 中写好的核心逻辑
from .ai_bot import (
    get_bilingual_response,
    get_relevant_context,
    rewrite_question,
    generate_summary,
    compress_summary,
    format_summary_for_prompt,
    SUMMARY_CONFIG
)

from django.shortcuts import render
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication


def chat_page(request):
    """当用户访问首页时，返回聊天网页"""
    return render(request, 'chat.html')


@api_view(['POST'])
def register_user(request):
    """接收用户名和密码，注册新用户"""
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "🎉 注册成功！欢迎加入。"}, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def chat_with_ai(request):
    user = request.user

    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    user_content = request.data.get('content')
    conversation_id = request.data.get('conversation_id')

    if not user_content:
        return Response({"error": "提问内容不能为空"}, status=status.HTTP_400_BAD_REQUEST)

    # 校验或创建会话
    if conversation_id:
        try:
            conversation = Conversation.objects.get(id=conversation_id, user=user)
        except Conversation.DoesNotExist:
            return Response({"error": "找不到该对话或您无权访问"}, status=status.HTTP_403_FORBIDDEN)
    else:
        conversation = Conversation.objects.create(
            title=user_content[:15] + "...",
            user=user
        )

    # ==========================================
    # 🚀 新增：长期记忆摘要机制
    # ==========================================
    # 获取对话总消息数（包括即将存入的用户消息）
    total_messages = Message.objects.filter(conversation=conversation).count() + 1

    # 摘要触发逻辑
    SUMMARY_THRESHOLD = SUMMARY_CONFIG["initial_threshold"]
    SUMMARY_UPDATE_INTERVAL = SUMMARY_CONFIG["update_interval"]

    summary_text = ""

    if total_messages >= SUMMARY_THRESHOLD:
        # 检查是否需要生成/更新摘要
        try:
            summary_obj = ConversationSummary.objects.get(conversation=conversation)
            last_summarized_id = summary_obj.last_summarized_message_id

            # 检查是否有足够的新消息需要摘要
            new_messages_count = Message.objects.filter(
                conversation=conversation,
                id__gt=last_summarized_id
            ).count()

            if new_messages_count >= SUMMARY_UPDATE_INTERVAL:
                # 更新摘要
                print(f"📝 [摘要更新] 对话 {conversation.id} 有 {new_messages_count} 条新消息，开始更新摘要...")

                new_messages = Message.objects.filter(
                    conversation=conversation,
                    id__gt=last_summarized_id
                ).order_by('created_at')

                summary_dict = generate_summary(
                    list(new_messages),
                    summary_obj.content
                )

                # 压缩检查
                if SUMMARY_CONFIG["compression_enabled"]:
                    summary_dict = compress_summary(
                        summary_dict,
                        SUMMARY_CONFIG["max_summary_length"]
                    )

                # 更新数据库
                summary_obj.content = json.dumps(summary_dict, ensure_ascii=False)
                summary_obj.last_summarized_message_id = new_messages.last().id
                summary_obj.save()

                summary_text = format_summary_for_prompt(summary_dict)
                print(f"✅ [摘要更新] 摘要已更新，覆盖至消息 ID {summary_obj.last_summarized_message_id}")
            else:
                # 使用现有摘要
                summary_dict = json.loads(summary_obj.content)
                summary_text = format_summary_for_prompt(summary_dict)
                print(f"📖 [摘要读取] 使用现有摘要（覆盖至消息 ID {last_summarized_id}）")

        except ConversationSummary.DoesNotExist:
            # 首次生成摘要
            print(f"🆕 [摘要生成] 对话 {conversation.id} 达到 {total_messages} 条消息，首次生成摘要...")

            # 获取窗口外的所有历史消息（排除最近 6 条）
            old_messages = Message.objects.filter(
                conversation=conversation
            ).order_by('created_at')

            # 如果消息数 > 6，则对前面的消息生成摘要
            if old_messages.count() > 6:
                messages_to_summarize = list(old_messages[:-6])

                summary_dict = generate_summary(messages_to_summarize)

                # 创建摘要记录
                ConversationSummary.objects.create(
                    conversation=conversation,
                    content=json.dumps(summary_dict, ensure_ascii=False),
                    last_summarized_message_id=messages_to_summarize[-1].id
                )

                summary_text = format_summary_for_prompt(summary_dict)
                print(f"✅ [摘要生成] 首次摘要已生成，覆盖 {len(messages_to_summarize)} 条历史消息")

    # ==========================================
    # 🚀 亮点：滑动窗口记忆（最近 3 轮 / 6 条消息）
    # ==========================================
    # 1. 按时间倒序拿到最近的 6 条记录（转化成 list 以便后续操作）
    recent_messages = list(Message.objects.filter(conversation=conversation).order_by('-created_at')[:6])

    # 2. 将这 6 条记录反转回正序，供大模型按正常时间线阅读
    history_messages = reversed(recent_messages)

    history_str = ""
    if summary_text:
        history_str = f"【长期记忆摘要】\n{summary_text}\n\n【最近对话】\n"

    for msg in history_messages:
        role_name = "用户" if msg.role == "user" else "AI"
        history_str += f"{role_name}: {msg.content}\n"
    # ==========================================

    # 存入用户的原始提问
    user_message = Message.objects.create(conversation=conversation, role='user', content=user_content)

    # 第 1 步：调用手写的重写函数（解决代词和话题漂移）
    standalone_question = rewrite_question(history_str, user_content)

    # 终端日志调试
    print(f"\n[后台路由] 当前会话 ID: {conversation.id}")
    print(f"[后台路由] 用户原提问: {user_content}")
    print(f"[后台路由] 大模型重写后搜索词: {standalone_question}\n")

    # 第 2 步：拿着重写后的清洗问题去查向量库
    retrieved_context = get_relevant_context(standalone_question, user)

    # 第 3 步：最终生成回答
    try:
        ai_response_content = get_bilingual_response(
            message=user_content,
            history=history_str,
            context=retrieved_context
        )
    except Exception as e:
        return Response({"error": f"大模型调用失败: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 存入 AI 的回答
    ai_message = Message.objects.create(conversation=conversation, role='assistant', content=ai_response_content)

    return Response({
        "conversation_id": conversation.id,
        "user_message": MessageSerializer(user_message).data,
        "ai_message": MessageSerializer(ai_message).data
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_conversation_list(request):
    """获取当前登录用户的所有历史会话列表"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    convs = Conversation.objects.filter(user=user).order_by('-created_at')
    res_data = [{"id": c.id, "title": c.title, "created_at": c.created_at.strftime("%m-%d %H:%M")} for c in convs]
    return Response(res_data, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_conversation_detail(request, conv_id):
    """获取某个具体会话的所有聊天记录"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        conv = Conversation.objects.get(id=conv_id, user=user)
        msgs = Message.objects.filter(conversation=conv).order_by('created_at')
        return Response(MessageSerializer(msgs, many=True).data, status=status.HTTP_200_OK)
    except Conversation.DoesNotExist:
        return Response({"error": "找不到该对话或无权访问"}, status=status.HTTP_403_FORBIDDEN)


@api_view(['DELETE'])
def delete_conversation(request, conv_id):
    """删除指定的历史会话"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        conv = Conversation.objects.get(id=conv_id, user=user)
        conv.delete()
        return Response({"message": "删除成功"}, status=status.HTTP_200_OK)
    except Conversation.DoesNotExist:
        return Response({"error": "找不到该对话或无权删除"}, status=status.HTTP_403_FORBIDDEN)


@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@parser_classes([MultiPartParser])
def upload_knowledge_base(request):
    """上传 txt/pdf 文件，解析并向量化为当前用户的临时知识库。"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    file = request.FILES.get('file')
    if not file:
        return Response({"error": "请上传文件"}, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(file.name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return Response({"error": f"仅支持 .txt 和 .pdf 文件"}, status=status.HTTP_400_BAD_REQUEST)

    if file.size > MAX_FILE_SIZE:
        return Response({"error": "文件大小不能超过 20MB"}, status=status.HTTP_400_BAD_REQUEST)

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        for chunk in file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        result = build_knowledge_base(tmp_path, user.id)
    except Exception as e:
        os.unlink(tmp_path)
        return Response({"error": f"知识库构建失败: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 先停用旧知识库并删除其文件
    old_kb = KnowledgeBaseSession.objects.filter(user=user, is_active=True).first()
    old_kb_dir = None
    if old_kb:
        old_kb_dir = os.path.dirname(old_kb.persist_directory)
        old_kb.is_active = False
        old_kb.save()

    # 创建新知识库记录
    new_kb = KnowledgeBaseSession.objects.create(
        user=user,
        name=file.name,
        source_file_name=file.name,
        collection_name=result["collection_name"],
        persist_directory=result["persist_directory"],
        bm25_index_path=result["bm25_index_path"],
        source_file_path=result["source_file_path"],
        chunk_count=result["chunk_count"],
        is_active=True,
    )

    # 删除旧目录（新库已激活后再删）
    if old_kb_dir:
        delete_knowledge_base_dir(old_kb_dir)

    return Response({
        "name": new_kb.name,
        "chunk_count": new_kb.chunk_count,
        "replaced": old_kb is not None,
    }, status=status.HTTP_201_CREATED)


@api_view(['GET', 'DELETE'])
@authentication_classes([JWTAuthentication])
def current_knowledge_base(request):
    """GET: 查询当前激活知识库；DELETE: 清空当前知识库。"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    kb = KnowledgeBaseSession.objects.filter(user=user, is_active=True).first()

    if request.method == 'GET':
        if not kb:
            return Response({"active": False}, status=status.HTTP_200_OK)
        return Response({
            "active": True,
            "name": kb.name,
            "chunk_count": kb.chunk_count,
            "created_at": kb.created_at.strftime("%m-%d %H:%M"),
        }, status=status.HTTP_200_OK)

    # DELETE
    if kb:
        kb_dir = os.path.dirname(kb.persist_directory)
        kb.delete()
        delete_knowledge_base_dir(kb_dir)
    return Response({"message": "知识库已清空"}, status=status.HTTP_200_OK)


@api_view(['GET'])
@authentication_classes([JWTAuthentication])
def knowledge_base_content(request):
    """返回当前激活知识库的源文件文本内容。"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    kb = KnowledgeBaseSession.objects.filter(user=user, is_active=True).first()
    if not kb:
        return Response({"error": "当前没有激活的知识库"}, status=status.HTTP_404_NOT_FOUND)

    file_path = kb.source_file_path
    if not file_path or not os.path.exists(file_path):
        return Response({"error": "源文件不存在"}, status=status.HTTP_404_NOT_FOUND)

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif ext == '.pdf':
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            content = '\n\n'.join([f"── 第 {i+1} 页 ──\n{doc.page_content}" for i, doc in enumerate(docs)])
        else:
            content = "不支持预览此文件类型"
    except Exception as e:
        return Response({"error": f"读取文件失败: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({
        "name": kb.name,
        "file_type": ext,
        "chunk_count": kb.chunk_count,
        "content": content,
    }, status=status.HTTP_200_OK)


@api_view(['GET', 'PUT'])
@authentication_classes([JWTAuthentication])
def user_profile(request):
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    profile, _ = UserProfile.objects.get_or_create(user=user)

    if request.method == 'GET':
        return Response({
            "username": user.username,
            "nickname": profile.nickname or user.username,
            "avatar": request.build_absolute_uri(profile.avatar.url) if profile.avatar else None,
        })

    # PUT: 更新昵称和/或密码
    nickname = request.data.get('nickname', '').strip()
    old_password = request.data.get('old_password', '')
    new_password = request.data.get('new_password', '')

    if nickname:
        profile.nickname = nickname
        profile.save()

    if new_password:
        if not user.check_password(old_password):
            return Response({"error": "旧密码错误"}, status=status.HTTP_400_BAD_REQUEST)
        user.set_password(new_password)
        user.save()

    return Response({"message": "保存成功"})


@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@parser_classes([MultiPartParser])
def upload_avatar(request):
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    file = request.FILES.get('avatar')
    if not file:
        return Response({"error": "请上传图片"}, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(file.name)[1].lower()
    if ext not in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
        return Response({"error": "仅支持 jpg/png/gif/webp"}, status=status.HTTP_400_BAD_REQUEST)

    if file.size > 5 * 1024 * 1024:
        return Response({"error": "图片不能超过 5MB"}, status=status.HTTP_400_BAD_REQUEST)

    profile, _ = UserProfile.objects.get_or_create(user=user)
    profile.avatar = file
    profile.save()

    return Response({"avatar": request.build_absolute_uri(profile.avatar.url)})