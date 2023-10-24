from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from common.models import Profile
from django import forms
from django.contrib.auth.forms import PasswordChangeForm
from django.utils.translation import gettext_lazy as _

class UserForm(UserCreationForm):
    email = forms.EmailField(label="이메일")
    class Meta:
        model = User
        fields = ("username", "email")

# Adding new form for introduction
class IntroForm(forms.Form):
    intro = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'placeholder': '여기에 한줄소개를 작성하세요'}))

class CustomPasswordChangeForm(PasswordChangeForm):
    old_password = forms.CharField(
        label=_("기존 비밀번호"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'current-password', 'autofocus': True}),
    )
    new_password1 = forms.CharField(
        label=_("새 비밀번호"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
    )
    new_password2 = forms.CharField(
        label=_("새 비밀번호 (확인)"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
    )

