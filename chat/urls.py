from django.urls import path
from . import views

urlpatterns = [
    # 接口地址将会是：.../api/chat/ask/
    path('ask/', views.chat_with_ai, name='chat_with_ai'),
]