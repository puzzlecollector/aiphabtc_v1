from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404, redirect
from ..models import Question, Board
from django.db.models import Q, Count


def index(request, board_name="perceptive"):
    # If directly accessing the main URL, redirect to the perceptive board
    if not board_name:
        return redirect('pybo:board_view', board_name='perceptive')

    board = get_object_or_404(Board, name=board_name)
    question_list = Question.objects.filter(board=board)

    page = request.GET.get("page", "1")  # page
    kw = request.GET.get('kw', '')  # 검색어
    so = request.GET.get("so", "recent")  # sort order

    # 정렬
    if so == "recommend":
        question_list = question_list.annotate(
            num_voter=Count('voter')).order_by("-num_voter", "-create_date")
    elif so == "popular":
        question_list = question_list.annotate(
            num_answer=Count('answer')).order_by("-num_answer", "-create_date")
    else:  # recent
        question_list = question_list.order_by("-create_date")

    # 조회
    if kw:
        question_list = question_list.filter(
            Q(subject__icontains=kw) |
            Q(content__icontains=kw) |
            Q(author__username__icontains=kw) |
            Q(answer__author__username__icontains=kw)
        ).distinct()

    # 페이징 처리
    paginator = Paginator(question_list, 10)
    page_obj = paginator.get_page(page)

    context = {"question_list": page_obj, "page": page, "kw": kw, "board_name": board_name}
    return render(request, 'pybo/question_list.html', context)


def detail(request, question_id):
    ''' Print out question content '''
    question = get_object_or_404(Question, pk=question_id)
    context = {"question": question}
    return render(request, "pybo/question_detail.html", context)
