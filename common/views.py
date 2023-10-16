from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from common.forms import UserForm, IntroForm
from common.models import Profile
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.files.images import get_image_dimensions

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

@login_required(login_url='common:login')
def account_page(request):
    user = request.user
    context = {
        'user': user,
        'email': user.email,
        'username': user.username,
        'intro': user.profile.intro,
    }
    return render(request, 'common/account_page.html', context)


@login_required(login_url='common:login')
def profile_modify_image(request):
    if request.method == "POST" and 'profile_picture' in request.FILES:
        profile_picture = request.FILES["profile_picture"]
        try:
            if not profile_picture.name.endswith(('.png', '.jpg', '.jpeg')):
                raise ValidationError("Invalid file type: Accepted file types are .png, .jpg, .jpeg")
            width, height = get_image_dimensions(profile_picture)
            max_dimensions = 800
            if width > max_dimensions or height > max_dimensions:
                raise ValidationError("Invalid image size: Max dimensions are 800x800px")
            # Save image to user's profile
            user_profile = request.user.profile  # assumes a related_name of 'profile'
            user_profile.image = profile_picture
            user_profile.save()

            messages.success(request, "Profile picture updated successfully!")
            return redirect('common:settings_base')  # Redirect to a different view after success
        except ValidationError as e:
            messages.error(request, f"Upload failed: {str(e)}")
    elif request.method == "POST":
        messages.error(request, "Something went wrong.")

    return render(request, 'common/settings/profile_picture.html')  # Render the template for image upload


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