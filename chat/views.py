from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Conversation, Message
from .serializers import MessageSerializer
# 新增导入 get_relevant_context
from .ai_bot import get_bilingual_response, get_relevant_context

from django.shortcuts import render

def chat_page(request):
    """当用户访问首页时，返回聊天网页"""
    return render(request, 'chat.html')


@api_view(['POST'])
def chat_with_ai(request):
    user_content = request.data.get('content')
    conversation_id = request.data.get('conversation_id')

    if not user_content:
        return Response({"error": "提问内容不能为空"}, status=status.HTTP_400_BAD_REQUEST)

    # 1. & 2. & 3. 处理对话 Session 并保存用户的提问 (这部分逻辑与之前完全一样)
    if conversation_id:
        conversation = Conversation.objects.get(id=conversation_id)
    else:
        conversation = Conversation.objects.create(title=user_content[:15] + "...")

    history_messages = Message.objects.filter(conversation=conversation).order_by('created_at')
    history_str = ""
    for msg in history_messages:
        role_name = "用户" if msg.role == "user" else "AI"
        history_str += f"{role_name}: {msg.content}\n"

    user_message = Message.objects.create(
        conversation=conversation,
        role='user',
        content=user_content
    )

    # ================= 核心 RAG 升级区 =================

    # 4. RAG 检索：拿着用户的问题，去向量数据库找答案！
    retrieved_context = get_relevant_context(user_content)

    # 5. 召唤大模型！把“历史聊天”、“查到的知识”、“新问题”三管齐下喂给它
    try:
        ai_response_content = get_bilingual_response(
            message=user_content,
            history=history_str,
            context=retrieved_context  # <--- 关键！知识被注入了！
        )
    except Exception as e:
        return Response({"error": f"大模型调用失败: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ===================================================

    # 6. 保存 AI 回答并返回 JSON (与之前完全一样)
    ai_message = Message.objects.create(
        conversation=conversation,
        role='assistant',
        content=ai_response_content
    )

    return Response({
        "conversation_id": conversation.id,
        "user_message": MessageSerializer(user_message).data,
        "ai_message": MessageSerializer(ai_message).data
    }, status=status.HTTP_200_OK)