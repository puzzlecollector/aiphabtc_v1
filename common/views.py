from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from common.forms import UserForm, IntroForm
from common.models import Profile, Attendance
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.files.images import get_image_dimensions
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from common.forms import CustomPasswordChangeForm
from django.utils import timezone
from datetime import timedelta
from django.utils.timezone import localtime


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

def get_tier(rank, total_users):
    percentile = (rank / total_users) * 100
    if percentile <= 0.01:
        return 'grandmaster', 'grandmaster.png'
    elif percentile <= 0.1:
        return 'challenger', 'challenger.png'
    elif percentile <= 1:
        return 'gold', 'gold.png'
    elif percentile <= 10:
        return 'silver', 'silver.png'
    elif percentile <= 20:
        return 'bronze', 'bronze.png'
    else:
        return 'beginner', 'beginner.png'

def ranking(request):
    user_list = Profile.objects.order_by('-score')
    total_users = user_list.count()
    page = request.GET.get('page', 1)
    paginator = Paginator(user_list, 20)

    try:
        profiles = paginator.page(page)
    except PageNotAnInteger:
        profiles = paginator.page(1)
    except EmptyPage:
        profiles = paginator.page(paginator.num_pages)

    profile_list_with_tier = []
    for index, profile in enumerate(user_list):
        rank = index + 1
        tier, icon = get_tier(rank, total_users)
        if profile in profiles.object_list:
            profile_list_with_tier.append((profile, rank, tier, icon))

    context = {
        'profile_list_with_tier': profile_list_with_tier,
    }
    return render(request, 'common/ranking.html', context)

def point_policy(request):
    return render(request, 'common/point_policy.html', {})

@login_required(login_url="common:login")
def password_reset(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, '비밀번호가 정상적으로 변경되었습니다.')
            return redirect('common:settings_base')
    else:
        form = CustomPasswordChangeForm(request.user)
    return render(request, 'common/settings/password_reset.html', {'form': form})


@login_required(login_url='common:login')
def attendance(request):
    user = request.user
    now = timezone.now()
    local_now = localtime(now)  # Convert to local timezone

    # Check if the user has already attended today
    attended_today = user.attendances.filter(timestamp__date=local_now.date()).exists()

    # Check the number of attendances in the last 30 days
    attendances_last_month = user.attendances.filter(timestamp__gte=now - timedelta(days=30)).count()

    if request.method == "POST" and not attended_today and attendances_last_month < 25:
        Attendance.objects.create(user=user)
        # Give the user 2 tokens (update your logic accordingly)
        user.profile.tokens += 2
        user.profile.save()
        messages.success(request, '출석체크 완료! 2 토큰을 받았습니다.')
        return redirect('common:attendance')

    context = {
        'attended_today': attended_today,
        'attendances_last_month': attendances_last_month,
        'tokens': user.profile.tokens,
        'now': local_now,  # Pass local datetime to the template
    }
    return render(request, 'common/attendance.html', context)

