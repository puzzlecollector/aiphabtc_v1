import sys
import os

from django.shortcuts import render



def independent_indicator_view(request):
    return render(request, 'independent_indicator_views.html')
