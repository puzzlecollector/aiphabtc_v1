from django.shortcuts import render, get_object_or_404
from ..models import Question, Board
from django.core.paginator import Paginator

def component_list(request, board_name=None):
    if board_name:
        board = get_object_or_404(Board, name=board_name)
        question_list = Question.objects.filter(board=board)
    else:
        question_list = Question.objects.all()

    # Paginate the question_list just like in the index view
    page = request.GET.get("page", "1")
    paginator = Paginator(question_list, 10)
    page_obj = paginator.get_page(page)

    context = {
        'question_list': page_obj,  # Pass the Page object, not the QuerySet
        'board_name': board_name,
    }

    return render(request, 'pybo/question_list.html', context)
