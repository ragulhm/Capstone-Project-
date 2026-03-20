"""Web page URLs for template-rendered views."""
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from .views import DashboardPageView, HomePageView

urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path(
        'login/',
        LoginView.as_view(
            template_name='inference/pages/login.html',
            redirect_authenticated_user=True,
        ),
        name='login',
    ),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('dashboard/', DashboardPageView.as_view(), name='dashboard'),
]
