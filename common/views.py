from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from common.forms import UserForm, IntroForm
from common.models import Profile
from django.contrib import messages
from django.contrib.auth.decorators import login_required
@login_required(login_url='common:login')
def base(request):
    # account settings base page
    if request.method == "POST":
        form = IntroForm(request.POST)
        if form.is_valid():
            request.user.profile.intro = form.cleaned_data["intro"]
            request.user.profile.save()
            messages.success(request, '프로필이 정상적으로 업데이트 되었습니다!') # adding message
            return redirect("common:settings_base")
    else:
        form = IntroForm(initial={"intro":request.user.profile.intro})
    context = {'settings_type': 'base', 'form': form}
    return render(request, 'common/settings/base.html', context)

@login_required
def account_page(request):
    user = request.user
    context = {
        'user': user,
        'email': user.email,
        'username': user.username,
        'intro': user.profile.intro,
    }
    return render(request, 'common/account_page.html', context)

def signup(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
        username = form.cleaned_data.get("username")
        raw_password = form.cleaned_data.get("password1")
        user = authenticate(username=username, password=raw_password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, "Authentication failed. Please try again.")
    else:
        # GET request
        form = UserForm()
    return render(request, "common/signup.html", {"form": form})

def page_not_found(request, exception):
    return render(request, 'common/404.html', {})