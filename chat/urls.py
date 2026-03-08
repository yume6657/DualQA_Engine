from django.urls import path
from . import views
# 🌟 新增：导入 JWT 官方提供的发证(登录)和换证接口
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('ask/', views.chat_with_ai, name='chat_with_ai'),
    path('register/', views.register_user, name='register'),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('history/', views.get_conversation_list, name='history_list'),
    path('history/<int:conv_id>/', views.get_conversation_detail, name='history_detail'),
    path('history/<int:conv_id>/delete/', views.delete_conversation, name='delete_conversation'),
]