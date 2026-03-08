from rest_framework import serializers
from .models import Conversation, Message

from rest_framework import serializers
from django.contrib.auth.models import User # 导入官方用户表
from .models import Message

# ================= 新增：用户序列化器 =================
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password']
        # 面试细节：告诉 Django，密码字段只能用来写入（注册），绝对不能在读取时返回给前端
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        # 面试细节：创建用户时，绝不能直接 save()，必须调用 create_user 进行哈希加密
        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password']
        )
        return user
# ====================================================


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