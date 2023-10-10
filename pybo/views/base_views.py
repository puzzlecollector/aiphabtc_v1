from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from ..models import Question
from django.db.models import Q, Count

def index(request):
    page = request.GET.get("page", "1") # page
    kw = request.GET.get('kw', '') # 검색어
    so = request.GET.get("so", "recent")

    # 정렬
    if so == "recommend":
        question_list = Question.objects.annotate(
            num_voter=Count('voter')).order_by("-num_voter", "-create_date")
    elif so == "popular":
        question_list = Question.objects.annotate(
            num_answer=Count('answer')).order_by("-num_answer", "-create_date")
    else: # recent
        question_list = Question.objects.order_by("-create_date")

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
    context = {"question_list": page_obj, "page": page, "kw": kw}
    return render(request, 'pybo/question_list.html', context)
    # return HttpResponse("안녕하세요 알파비트에 온걸 환영합니다! 저희는 비트코인 가격 관점공유 플랫폼 입니다.")

def detail(request, question_id):
    '''' print out question content '''
    question = get_object_or_404(Question, pk=question_id)
    context = {"question": question}
    return render(request, "pybo/question_detail.html", context)