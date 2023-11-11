from django.shortcuts import render, get_object_or_404
from common.models import Profile, Question, Answer, Comment
from django.contrib.auth.models import User

def profile_detail(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    profile = get_object_or_404(Profile, user=user)

    # Fetch user's questions, answers, and comments
    user_questions = Question.objects.filter(author=user)
    user_answers = Answer.objects.filter(author=user)
    user_comments = Comment.objects.filter(author=user)

    context = {
        'profile': profile,
        'user_questions': user_questions,
        'user_answers': user_answers,
        'user_comments': user_comments
    }
    return render(request, "profile_detail.html", context)