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



@api_view(['GET'])
def get_conversation_list(request):
    """获取当前登录用户的所有历史会话列表"""
    user = request.user
    if not user.is_authenticated:
        return Response({"error": "请先登录"}, status=status.HTTP_401_UNAUTHORIZED)


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
        conv.delete()  # 触发级联删除，相关的 Message 也会被一并清空
        return Response({"message": "删除成功"}, status=status.HTTP_200_OK)
    except Conversation.DoesNotExist:
        return Response({"error": "找不到该对话或无权删除"}, status=status.HTTP_403_FORBIDDEN)