from django.db import models

# Create your models here.
from django.db import models

class Conversation(models.Model):
    """
    对话会话表：管理多轮对话的 Session
    """
    # CharField 对应 MySQL 里的 VARCHAR 类型
    title = models.CharField(max_length=255, blank=True, null=True, verbose_name="对话标题")


    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    updated_at = models.DateTimeField(auto_now=True, verbose_name="最后活跃时间")

    class Meta:
        db_table = 'chat_conversation' 
        verbose_name = '对话记录'
        verbose_name_plural = verbose_name


class Message(models.Model):
    """
    消息明细表：记录每一条具体的聊天
    """
    ROLE_CHOICES = (
        ('user', '用户提问'),
        ('assistant', 'AI回答'),
        ('system', '系统提示词'),
    )

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages',
                                     verbose_name="所属对话")

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, verbose_name="角色")
    content = models.TextField(verbose_name="内容")

    # 记录这句话是中文还是英文
    language = models.CharField(max_length=20, default='auto', verbose_name="语言类型")


    tokens_used = models.IntegerField(default=0, verbose_name="消耗Token数")
    # ------------------------------

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="发送时间")

    class Meta:
        db_table = 'chat_message'
        ordering = ['created_at']
        verbose_name = '消息明细'
        verbose_name_plural = verbose_name