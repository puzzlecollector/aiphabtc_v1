from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseNotFound
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from ..forms import QuestionForm
from ..models import Question, Board
import time
from datetime import timedelta

@login_required(login_url="common:login")
def question_create(request, board_name=None):
    if request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            subject = form.cleaned_data.get("subject")
            content = form.cleaned_data.get("content")
            question = form.save(commit=False)
            question.author = request.user
            if board_name:
                board = get_object_or_404(Board, name=board_name)
                question.board = board
            question.create_date = timezone.now()

            if board_name == "perceptive":
                word_count = len(content.split())
                if word_count < 55:
                    messages.error(request, "내용은 최소 55단어 이상이어야 합니다!")
                    return redirect('pybo:question_create', board_name=board_name)
                crypto = request.POST.get("crypto")
                duration = request.POST.get("duration")
                direction = request.POST.get("direction")
                price_change = request.POST.get("price_change")
                # Add validation for price_change based on direction
                try:
                    price_change_value = float(price_change.strip('%'))
                except ValueError:
                    messages.error(request, "올바르지 않은 가격 변동 값입니다!")
                    return redirect('pybo:question_create', board_name=board_name)
                if direction == "bullish" and price_change_value < 0:
                    messages.error(request, "상승 추세를 위한 가격 변동은 0% 이상이어야 합니다!")
                    return redirect('pybo:question_create', board_name=board_name)
                elif direction == "bearish" and price_change_value > 0:
                    messages.error(request, "하락 추세를 위한 가격 변동은 0% 이하이어야 합니다!")
                    return redirect('pybo:question_create', board_name=board_name)

                last_prediction = Question.objects.filter(
                    author=request.user,
                    subject__regex=rf"\[{crypto}\]\[{duration}\].*"
                ).order_by('-create_date').first()
                if last_prediction:
                    duration_hours = {
                        '1 hour': 1,
                        '2 hours': 2,
                        '4 hours': 4,
                        '6 hours': 6,
                        '8 hours': 8,
                        '12 hours': 12,
                        '24 hours': 24,
                        'one week': 24 * 7,
                        'one month': 24 * 30,
                        'one year': 24 * 365
                    }
                    end_time = last_prediction.create_date + timedelta(
                        hours=duration_hours[duration])
                    if timezone.now() < end_time:
                        messages.error(request,
                                       '해당 코인에 대해서 이 기간 동안 이전 예측이 끝나기 전까지 다른 예측을 게시할 수 없습니다!')
                        return redirect('pybo:question_create', board_name=board_name)
                    else:
                        subject = f"[{crypto}][{duration}][{direction}][{price_change}] {subject}"
                        question.subject = subject
                        question.content = content
                else:
                    subject = f"[{crypto}][{duration}][{direction}][{price_change}] {subject}"
                    question.subject = subject
                    question.content = content
            elif board_name == "technical_blog" or board_name == "trading_blog":
                word_count = len(content.split())
                if word_count < 55:
                    messages.error(request, "내용은 최소 55단어 이상이어야 합니다!")
                    return redirect('pybo:question_create', board_name=board_name)

            question.save()

            if board_name:
                return redirect('pybo:board_view', board_name=board_name)
            else:
                return redirect("pybo:index")
    else:
        form = QuestionForm()
    context = {"form": form}
    if board_name == "perceptive":
        context["is_perceptive_board"] = True
    return render(request, "pybo/question_form.html", context)


@login_required(login_url = "common:login")
def question_modify(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, "수정권한이 없습니다!")
        return redirect('pybo:detail', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.author = request.user
            question.modify_date = timezone.now()
            content = form.cleaned_data.get("content")
            if question.board.name == "perceptive" or question.board.name == "technical_blog" or question.board.name == "trading_blog":
                # Validate word count for content
                word_count = len(content.split())
                if word_count < 55:
                    messages.error(request, "내용은 최소 55단어 이상이어야 합니다!")
                    return redirect('pybo:question_modify', question_id=question.id)
            question.save()
            return redirect('pybo:detail', question_id=question.id)
    else:
        form = QuestionForm(instance=question)
    context = {'form': form}
    return render(request, 'pybo/question_form.html', context)

@login_required(login_url='common:login')
def question_delete(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, "삭제권한이 없습니다!")
        return redirect('pybo:detail', question_id=question.id)
    question.delete()
    return redirect('pybo:index')