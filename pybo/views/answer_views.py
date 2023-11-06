from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect, resolve_url
from django.utils import timezone
from ..forms import AnswerForm
from ..models import Question, Answer
from common.models import PointTokenTransaction
@login_required(login_url="common:login")

def answer_create(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == "POST":
        form = AnswerForm(request.POST)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.author = request.user # 추가한 속성 author 적용
            answer.create_date = timezone.now()
            answer.question = question
            answer.save()
            if answer.question.board.name == "perceptive":
                points_for_question = 4
            elif answer.question.board.name in ["free_board", "technical_blog", "trading_blog"]:
                points_for_question = 2
            eng_to_kor_name = {'perceptive': '관점공유 게시판', 'free_board': '자유게시판', 'technical_blog': '기술 블로그',
                               'trading_blog': '트레이딩 블로그'}
            kor_board_name = eng_to_kor_name[answer.question.board.name]
            PointTokenTransaction.objects.create(
                user=request.user,
                points=points_for_question,
                tokens=0,  # Assuming you only want to deal with points
                reason=f"{kor_board_name}에 답글 포스팅"
            )
            return redirect('{}#answer_{}'.format(resolve_url('pybo:detail', question_id=question.id), answer.id))
            # return redirect("pybo:detail", question_id=question_id)
    else:
        form = AnswerForm()
    context = {"question": question, "form": form}
    return render(request, "pybo/question_detail.html", context)

@login_required(login_url="common:login")
def answer_modify(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, "수정권한이 없습니다!")
        return redirect("pybo:detail", question_id=answer.question.id)
    if request.method == "POST":
        form = AnswerForm(request.POST, instance=answer)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.author = request.user
            answer.modify_date = timezone.now()
            answer.save()
            return redirect('{}#answer_{}'.format(resolve_url('pybo:detail', question_id=answer.question.id), answer.id))
            # return redirect('pybo:detail', question_id=answer.question.id)
    else:
        form = AnswerForm(instance=answer)
    context = {"answer":answer, "form":form}
    return render(request, "pybo/answer_form.html", context)

@login_required(login_url="common:login")
def answer_delete(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, "삭제권한이 없습니다!")
    else:
        answer.delete()
        if answer.question.board.name == "perceptive":
            points_for_question = -4
        elif answer.question.board.name in ["free_board", "technical_blog", "trading_blog"]:
            points_for_question = -2
        eng_to_kor_name = {'perceptive': '관점공유 게시판', 'free_board': '자유게시판', 'technical_blog': '기술 블로그',
                           'trading_blog': '트레이딩 블로그'}
        kor_board_name = eng_to_kor_name[answer.question.board.name]
        PointTokenTransaction.objects.create(
            user=request.user,
            points=points_for_question,
            tokens=0,  # Assuming you only want to deal with points
            reason=f"{kor_board_name}에 답글 삭제"
        )
    return redirect("pybo:detail", question_id=answer.question.id)
