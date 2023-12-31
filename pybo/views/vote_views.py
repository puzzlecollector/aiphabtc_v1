from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect

from ..models import Question, Answer
from common.models import Profile, PointTokenTransaction

@login_required(login_url='common:login')
def vote_question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user == question.author:
        messages.error(request, '본인의 글은 추천할 수 없어요!')
    else:
        if request.user not in question.voter.all():
            question.voter.add(request.user)
            # Update scores
            author_profile = question.author.profile
            author_profile.score += 1  # author receives 2 points for a like
            author_profile.save()
            PointTokenTransaction.objects.create(
                user=question.author,
                points=2,
                tokens=0,
                reason="{}님이 당신의 질문을 추천했습니다!".format(request.user.username)
            )

            user_profile = request.user.profile
            user_profile.score += 1  # user gives 1 point for a like
            user_profile.save()
            PointTokenTransaction.objects.create(
                user=request.user,
                points=1,
                tokens=0,
                reason="질문 추천 참여"
            )
        else:
            messages.error(request, '이미 추천한 글입니다.')

    return redirect('pybo:detail', question_id=question.id)

@login_required(login_url='common:login')
def vote_answer(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user == answer.author:
        messages.error(request, '본인이 작성한 글은 추천할 수 없습니다.')
    else:
        if request.user not in answer.voter.all():
            answer.voter.add(request.user)
            # Update scores
            author_profile = answer.author.profile
            author_profile.score += 1  # author receives 2 points for a like
            author_profile.save()
            PointTokenTransaction.objects.create(
                user=answer.author,
                points=2,
                tokens=0,
                reason="{}님이 당신의 답변을 추천했습니다!".format(request.user.username)
            )

            user_profile = request.user.profile
            user_profile.score += 1  # user gives 1 point for a like
            user_profile.save()
            PointTokenTransaction.objects.create(
                user=request.user,
                points=1,
                tokens=0,
                reason="답변 추천 참여"
            )
        else:
            messages.error(request, '이미 추천한 글입니다.')

    return redirect('pybo:detail', question_id=answer.question.id)

