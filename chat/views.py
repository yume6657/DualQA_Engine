from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Conversation, Message
from .serializers import MessageSerializer, UserSerializer
# 新增导入 get_relevant_context
from .ai_bot import get_bilingual_response, get_relevant_context

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
        serializer.save() # 这里会自动触发我们在 Serializer 里写的密码加密逻辑
        return Response({"message": "🎉 注册成功！欢迎加入。"}, status=status.HTTP_201_CREATED)
    # 如果用户名已被注册，或者格式不对，直接把报错信息扔给前端
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)





@api_view(['POST'])
# 🌟 核心：这两行装饰器告诉 Django，访问这个接口必须带上合法的令牌
def chat_with_ai(request):
    # 1. 自动识别用户：只要令牌合法，Django 会自动把用户对象塞进 request.user
    user = request.user

    # 验证用户是否已通过认证 (虽然有装饰器，但写个判断更严谨)
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    user_content = request.data.get('content')
    conversation_id = request.data.get('conversation_id')

    if not user_content:
        return Response({"error": "提问内容不能为空"}, status=status.HTTP_400_BAD_REQUEST)

    # 2. 对话隔离：查询或创建对话时，必须带上 user=user
    if conversation_id:
        try:
            # 只能获取属于当前用户的对话，防止越权查看他人记录
            conversation = Conversation.objects.get(id=conversation_id, user=user)
        except Conversation.DoesNotExist:
            return Response({"error": "找不到该对话或您无权访问"}, status=status.HTTP_403_FORBIDDEN)
    else:
        # 新建对话时，自动绑定到当前登录的用户
        conversation = Conversation.objects.create(
            title=user_content[:15] + "...",
            user=user
        )

    # 3. 后续逻辑保持不变（获取历史、检索知识、调 AI）
    history_messages = Message.objects.filter(conversation=conversation).order_by('created_at')
    history_str = ""
    for msg in history_messages:
        role_name = "用户" if msg.role == "user" else "AI"
        history_str += f"{role_name}: {msg.content}\n"

    user_message = Message.objects.create(conversation=conversation, role='user', content=user_content)

    retrieved_context = get_relevant_context(user_content)

    try:
        ai_response_content = get_bilingual_response(
            message=user_content,
            history=history_str,
            context=retrieved_context
        )
    except Exception as e:
        return Response({"error": f"大模型调用失败: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    ai_message = Message.objects.create(conversation=conversation, role='assistant', content=ai_response_content)

    return Response({
        "conversation_id": conversation.id,
        "user_message": MessageSerializer(user_message).data,
        "ai_message": MessageSerializer(ai_message).data
    }, status=status.HTTP_200_OK)


# ================= 新增：历史记录接口 =================

@api_view(['GET'])
def get_conversation_list(request):
    """获取当前登录用户的所有历史会话列表"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    # 面试考点：只查询属于当前 user 的数据，按时间倒序排列
    convs = Conversation.objects.filter(user=user).order_by('-created_at')

    # 组装简单的数据返回给前端的侧边栏
    res_data = [{"id": c.id, "title": c.title, "created_at": c.created_at.strftime("%m-%d %H:%M")} for c in convs]
    return Response(res_data, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_conversation_detail(request, conv_id):
    """获取某个具体会话的所有聊天记录"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        # 面试考点：必须同时校验 conv_id 和 user，防止用户A偷看用户B的记录 (越权访问)
        conv = Conversation.objects.get(id=conv_id, user=user)
        msgs = Message.objects.filter(conversation=conv).order_by('created_at')

        # 复用之前的 MessageSerializer 序列化数据
        return Response(MessageSerializer(msgs, many=True).data, status=status.HTTP_200_OK)
    except Conversation.DoesNotExist:
        return Response({"error": "找不到该对话或无权访问"}, status=status.HTTP_403_FORBIDDEN)


# ================= 新增：删除对话接口 =================
@api_view(['DELETE'])
def delete_conversation(request, conv_id):
    """删除指定的历史会话"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)

    try:

        conv = Conversation.objects.get(id=conv_id, user=user)
        conv.delete()  # 触发级联删除，相关的 Message 也会被一并清空
        return Response({"message": "删除成功"}, status=status.HTTP_200_OK)
    except Conversation.DoesNotExist:
        return Response({"error": "找不到该对话或无权删除"}, status=status.HTTP_403_FORBIDDEN)