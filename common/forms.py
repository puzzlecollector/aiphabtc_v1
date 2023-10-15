from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class UserForm(UserCreationForm):
    email = forms.EmailField(label="이메일")
    class Meta:
        model = User
        fields = ("username", "email")

# Adding new form for introduction
class IntroForm(forms.Form):
    intro = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'placeholder': '여기에 한줄소개를 작성하세요'}))
