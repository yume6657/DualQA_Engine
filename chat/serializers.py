from rest_framework import serializers
from .models import Conversation, Message

from rest_framework import serializers
from django.contrib.auth.models import User # 导入官方用户表
from .models import Message


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password']

        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):

        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password']
        )
        return user


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