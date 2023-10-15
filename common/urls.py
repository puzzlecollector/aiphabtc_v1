from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
app_name = 'common'

urlpatterns = [
    path("login/", auth_views.LoginView.as_view(template_name="common/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("signup/", views.signup, name="signup"),

    # account settings
    path("settings/base/", views.base, name="settings_base"),
    # path("settings/password_change/", settings_views.PasswordChangeView.as_view(), name="password_change),
    # path("settings/image/", settings_views.profile_modify, name="settings_image"),
    # path("settings/image/delete/", settings_views.profile_image_delete, name="settings_image_delete"),

    # account page
    path("account_page/", views.account_page, name="account_page"),
]