from django.shortcuts import render

def free_board_view(request):
    return render(request, 'free_board_views.html')
