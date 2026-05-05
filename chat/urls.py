from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('ask/', views.chat_with_ai, name='chat_with_ai'),
    path('register/', views.register_user, name='register'),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('history/', views.get_conversation_list, name='history_list'),
    path('history/<int:conv_id>/', views.get_conversation_detail, name='history_detail'),
    path('history/<int:conv_id>/delete/', views.delete_conversation, name='delete_conversation'),
    path('knowledge-base/upload/', views.upload_knowledge_base, name='kb_upload'),
    path('knowledge-base/current/', views.current_knowledge_base, name='kb_current'),
    path('knowledge-base/content/', views.knowledge_base_content, name='kb_content'),
    path('profile/', views.user_profile, name='user_profile'),
    path('profile/avatar/', views.upload_avatar, name='upload_avatar'),
]