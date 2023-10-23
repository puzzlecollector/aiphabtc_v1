from django.urls import path
from .views import base_views, question_views, answer_views, comment_views, vote_views, free_board_views, indicator_views, profile_click_views, component_views, ai_indicator_views, trading_bot_views
app_name = "pybo"

urlpatterns = [
    # base_views.py
    path('',
         base_views.index, name="index"),
    path('<int:question_id>/',
         base_views.detail, name='detail'),

    # question_views.py
    #path('question/create/',
    #     question_views.question_create, name='question_create'),
    path('question/create/<str:board_name>/',
         question_views.question_create, name="question_create"),
    path('question/modify/<int:question_id>/',
         question_views.question_modify, name='question_modify'),
    path('question/delete/<int:question_id>/',
         question_views.question_delete, name='question_delete'),

    # answer_views.py
    path('answer/create/<int:question_id>/',
         answer_views.answer_create, name='answer_create'),
    path('answer/modify/<int:answer_id>/',
         answer_views.answer_modify, name='answer_modify'),
    path('answer/delete/<int:answer_id>/',
         answer_views.answer_delete, name='answer_delete'),

    # comment_views.py
    path('comment/create/questions/<int:question_id>/',
         comment_views.comment_create_question, name='comment_create_question'),
    path('comment/modify/question/<int:comment_id>/',
         comment_views.comment_modify_question, name='comment_modify_question'),
    path('comment/delete/question/<int:comment_id>/',
         comment_views.comment_delete_question, name='comment_delete_question'),
    path('comment/create/answer/<int:answer_id>/',
         comment_views.comment_create_answer, name='comment_create_answer'),
    path('comment/modify/answer/<int:comment_id>/',
         comment_views.comment_modify_answer, name='comment_modify_answer'),
    path('comment/delete/answer/<int:comment_id>/',
         comment_views.comment_delete_answer, name='comment_delete_answer'),

    # vote_views.py
    path('vote/question/<int:question_id>/',
         vote_views.vote_question, name='vote_question'),
    path('vote/answer/<int:answer_id>/',
         vote_views.vote_answer, name='vote_answer'),

    # free board
    path('free_board/',
         free_board_views.free_board_view, name='free_board'),
    path('indicator_page/',
         indicator_views.indicator_view, name='indicator_view'),
    path('ai_indicator_page/',
         ai_indicator_views.indicator_view, name='ai_indicator_view'),
    path('trading_bot_page/',
         trading_bot_views.bot_view, name='trading_bot_view'),

    # view other profiles
    path('profile/<int:user_id>/',
         profile_click_views.profile_detail, name='profile_detail'),

    # view other components
    path('board/<str:board_name>/',
         base_views.index, name='board_view'),
]
