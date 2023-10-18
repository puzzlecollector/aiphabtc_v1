from django.shortcuts import render, get_object_or_404
from common.models import Profile

def profile_detail(request, user_id):
    profile = get_object_or_404(Profile, user__id=user_id)
    return render(request, "profile_detail.html", {"profile": profile})