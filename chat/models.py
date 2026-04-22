from django.db import models
from django.contrib.auth.models import User  # 🌟 核心：导入 Django 官方的强大用户表


class Conversation(models.Model):
    # 🌟 建立关联：一条对话属于一个具体的用户
    # null=True, blank=True 是为了兼容你之前没绑定用户的旧测试数据
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - {self.user.username if self.user else '匿名'}"


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    role = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role}: {self.content[:20]}"


class ConversationSummary(models.Model):
    """对话长期记忆摘要"""
    conversation = models.OneToOneField(
        Conversation,
        on_delete=models.CASCADE,
        related_name='summary'
    )

    # 摘要内容（结构化 JSON 存储）
    content = models.TextField(help_text="JSON 格式存储的结构化摘要")

    # 摘要覆盖的消息范围
    last_summarized_message_id = models.BigIntegerField(
        help_text="最后一条被纳入摘要的消息 ID"
    )

    # 元数据
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Summary for {self.conversation.title}"