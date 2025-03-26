from django.urls import path
from django.contrib.auth import views as auth_views
from .views import get_csrf_token, get_user_data

urlpatterns = [
    path("auth/login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),  # ✅ Fix: Add login route
    path('auth/logout/', auth_views.LogoutView.as_view(), name='logout'),
    path("api/auth/user/", get_user_data, name="user-data"),
    path("csrf/", get_csrf_token, name="csrf-token"),
]

