from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from ..forms import QuestionForm
from ..models import Question, Board

@login_required(login_url="common:login")
def question_create(request, board_name=None):
    if request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.save(commit=False)
            question.author = request.user  # Added author attribute
            if board_name:
                board = get_object_or_404(Board, name=board_name)
                question.board = board  # Corrected the typo here
            question.create_date = timezone.now()
            question.save()

            # Redirect to the specific board's view after adding the question
            if board_name:
                return redirect('pybo:board_view', board_name=board_name)
            else:
                return redirect("pybo:index")

    else:
        form = QuestionForm()
    context = {"form": form}
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