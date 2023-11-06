from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from django.conf import settings
from django.conf.urls.static import static
app_name = 'common'

urlpatterns = [
    path("login/", auth_views.LoginView.as_view(template_name="common/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("signup/", views.signup, name="signup"),

    # account settings
    path("settings/base/", views.base, name="settings_base"),
    # path("settings/password_change/", settings_views.PasswordChangeView.as_view(), name="password_change),
    path("settings/image/", views.profile_modify_image, name="settings_image"),
    # path("settings/image/delete/", settings_views.profile_image_delete, name="settings_image_delete"),
    path('settings/password_reset/', views.password_reset, name='password_reset'),

    # account page
    path("account_page/", views.account_page, name="account_page"),

    # ranking page
    path('ranking/', views.ranking, name='ranking'),

    # point policy page
    path("point_policy/", views.point_policy, name="point_policy"),

    # attendance page
    path('attendance/', views.attendance, name='attendance'),

    path('transactions/<int:transaction_id>/', views.transaction_detail, name='transaction_detail')
]
