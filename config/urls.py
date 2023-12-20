"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from pybo.views import base_views
from django.conf import settings
from django.conf.urls.static import static
from pybo.views import indicator_views, ai_indicator_views, profile_click_views, independent_indicator_views, nlp_dashboard_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('pybo/', include('pybo.urls')),
    path('common/', include('common.urls')),
    path('', base_views.index, name='index'),
    # crypto fear greed index GPT response
    path('fetch_ai_analysis/',
         indicator_views.fetch_ai_analysis, name='fetch_ai_analysis'),
    path('fetch_ai_analysis_global/',
         indicator_views.fetch_ai_analysis_global, name='fetch_ai_analysis_global'),
    path('fetch_ai_technical/',
         indicator_views.fetch_ai_technical, name='fetch_ai_technical'),
    path('fetch_ai_corr/',
         ai_indicator_views.fetch_ai_corr, name='fetch_ai_corr'),
    path('fetch_forecast/',
         ai_indicator_views.time_series_views, name='fetch_forecast'),
    path('search_news/',
         ai_indicator_views.search_news, name='search_news'),
    path('fetch-time-series-analysis-indicator-page/<str:timeframe>/',
         independent_indicator_views.time_series_analysis, name='time_series_analysis'),
    path('fetch-gpt-analysis/<str:timeframe>/',
         independent_indicator_views.fetch_ai_analysis, name='fetch_ai_analysis'),
    path('news_similarity/',
         nlp_dashboard_views.search_news, name="nlp_dashboard_search_news"),
    path('search_chart_pattern/',
         nlp_dashboard_views.search_chart_pattern, name="nlp_dashboard_search_chart_patterns"),
]

handler404 = 'common.views.page_not_found'

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

