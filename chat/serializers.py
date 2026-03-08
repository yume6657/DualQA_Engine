from rest_framework import serializers
from .models import Conversation, Message

# =====================================================================
# 面试核心概念：序列化器 (Serializer)
# 作用：将 Django 的数据库对象翻译成纯粹的 JSON 数据，实现前后端分离。
# =====================================================================

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        # 返回给前端这几个关键字段
        fields = ['id', 'role', 'content', 'created_at']

class ConversationSerializer(serializers.ModelSerializer):
    # 嵌套序列化：查对话列表时，把里面的所有聊天记录也一并带出来
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'messages']