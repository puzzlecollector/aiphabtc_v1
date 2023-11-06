from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect, resolve_url
from django.utils import timezone
from ..forms import CommentForm
from ..models import Question, Answer, Comment
from common.models import PointTokenTransaction

@login_required(login_url='common:login')
def comment_create_question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == "POST":
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.author = request.user
            comment.create_date = timezone.now()
            comment.question = question
            comment.save()
            PointTokenTransaction.objects.create(
                user=request.user,
                points=1,
                tokens=0,  # Assuming you only want to deal with points
                reason="댓글 작성"
            )
            return redirect('pybo:detail', question_id=question_id)
    else:
        form = CommentForm()
    context = {'form':form}
    return render(request, 'pybo/comment_form.html', context)

@login_required(login_url="common:login")
def comment_modify_question(request, comment_id):
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, "댓글수정권한이 없습니다")
        return redirect('pybo:detail', question_id=comment.question.id)
    if request.method == "POST":
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.author = request.user
            comment.modify_date = timezone.now()
            comment.save()
            return redirect('pybo:detail', question_id=comment.question.id)
    else:
        form = CommentForm(instance=comment)
    context = {'form':form}
    return render(request, 'pybo/comment_form.html', context)

@login_required(login_url="common:login")
def comment_delete_question(request, comment_id):
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, '댓글삭제권한이 없습니다')
        return redirect('pybo:detail', question_id=comment.question.id)
    else:
        comment.delete()
        PointTokenTransaction.objects.create(
            user=request.user,
            points=-1,
            tokens=0,  # Assuming you only want to deal with points
            reason="댓글 삭제"
        )
    return redirect('pybo:detail', question_id=comment.question.id)

@login_required(login_url='common:login')
def comment_create_answer(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.method == "POST":
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.author = request.user
            comment.create_date = timezone.now()
            comment.answer = answer
            comment.save()
            PointTokenTransaction.objects.create(
                user=request.user,
                points=1,
                tokens=0,  # Assuming you only want to deal with points
                reason="댓글 작성"
            )
            return redirect("pybo:detail", question_id=comment.answer.question.id)
    else:
        form = CommentForm()
    context = {"form":form}
    return render(request, "pybo/comment_form.html", context)

@login_required(login_url='common:login')
def comment_modify_answer(request, comment_id):
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, "댓글수정권한이 없습니다")
        return redirect('pybo:detail', question_id=comment.answer.question.id)
    if request.method == "POST":
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.author = request.user
            comment.modify_date = timezone.now()
            comment.save()
            return redirect('pybo:detail', question_id=comment.answer.question.id)
    else:
        form = CommentForm(instance=comment)
    context = {'form':form}
    return render(request, 'pybo/comment_form.html', context)

@login_required(login_url="common:login")
def comment_delete_answer(request, comment_id):
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, "댓글삭제권한이 없습니다")
        return redirect('pybo:detail', question_id=comment.answer.question.id)
    else:
        comment.delete()
        PointTokenTransaction.objects.create(
            user=request.user,
            points=-1,
            tokens=0,  # Assuming you only want to deal with points
            reason="댓글 삭제"
        )
    return redirect('pybo:detail', question_id=comment.answer.question.id)