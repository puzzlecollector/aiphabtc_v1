from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from common.forms import UserForm, ProfileForm
from common.models import Profile, Attendance, PointTokenTransaction
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.files.images import get_image_dimensions
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from common.forms import CustomPasswordChangeForm
from django.utils import timezone
from datetime import timedelta
from django.utils.timezone import now, localtime

@login_required(login_url='common:login')
def base(request):
    # account settings base page
    if request.method == "POST":
        # Instantiate the form with the posted data and files (if there are any)
        form = ProfileForm(request.POST, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, '프로필이 정상적으로 업데이트 되었습니다!')
            return redirect("common:settings_base")
    else:
        # Instantiate the form with the current user's profile data
        form = ProfileForm(instance=request.user.profile)
    context = {'settings_type': 'base', 'form': form}
    return render(request, 'common/settings/base.html', context)

@login_required(login_url='common:login')
def account_page(request):
    user = request.user
    profile = user.profile
    context = {
        'user': user,
        'profile': profile,
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
    current_time = timezone.localtime(timezone.now())
    # calculate the number of attendances in the current month
    attendances_this_month = user.attendances.filter(
        timestamp__year = current_time.year,
        timestamp__month = current_time.month
    ).count()

    attended_today = user.attendances.filter(timestamp__date=current_time.date()).exists()

    if request.method == "POST" and not attended_today and attendances_this_month < 25:
        Attendance.objects.create(user=user, timestamp=current_time)
        user.profile.tokens += 2
        user.profile.save()

        PointTokenTransaction.objects.create(
            user=user,
            tokens=2,
            points=0,
            reason="출석체크 보상"
        )

        messages.success(request, "출석체크 완료! 2 토큰을 받았습니다.")
        return redirect('common:attendance')

    context = {
        'attended_today': attended_today,
        'attendances_this_month': attendances_this_month,
        'tokens': user.profile.tokens,
        'now': current_time,
    }
    return render(request, 'common/attendance.html', context)

@login_required(login_url='common:login')
def transaction_detail(request, transaction_id):
    transaction = get_object_or_404(PointTokenTransaction, id=transaction_id, user=request.user)
    transaction_list = PointTokenTransaction.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'common/transaction_detail.html', {'transaction':transaction, 'transaction_list':transaction_list})

def referral_view(request):
    if not request.user.is_authenticated:
        return redirect('common:login')
    user_profile = request.user.profile
    if request.method == "POST":
        referral_code = request.POST.get('referral_code').strip()
        # check if the user has already used a referral code
        if user_profile.referred_by is not None:
            messages.error(request, '이미 추천 코드를 사용하셨습니다.')
        # check if the referral code is the user's own code
        elif referral_code == user_profile.referral_code:
            messages.error(request, "본인의 레퍼럴 코드를 사용할 수 없어요!")
        elif referral_code:
            try:
                referrer_profile = Profile.objects.get(referral_code=referral_code)
                # additional check to prevent self referral
                if referrer_profile.user == request.user:
                    messages.error(request, '본인의 레퍼럴 코드를 사용할 수 없어요!')
                    return redirect('referral')
                user_profile.referred_by = referrer_profile
                user_profile.tokens += 50
                referrer_profile.tokens += 50
                user_profile.save()
                referrer_profile.save()
                PointTokenTransaction.objects.create(
                    user=request.user,
                    points=0,
                    tokens=50,
                    reason='레퍼럴 코드 입력'
                )
                PointTokenTransaction.objects.create(
                    user=referrer_profile.user,
                    points=0,
                    tokens=50,
                    reason="레퍼럴 코드 입력"
                )
                messages.success(request, '추천 코드가 승인되었습니다. 귀하와 추천인 모두 토큰을 받게 되었습니다.')
            except Profile.DoesNotExist:
                messages.error(request, '유효하지 않은 레퍼릴 코드입니다!')
    else:
        referral_code = None
    context = {
        "referral_code": user_profile.referral_code,
        "has_referred":  user_profile.referred_by is not None
    }
    return render(request, 'common/referral.html', context)