from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseNotFound
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone  # Correctly import Django's timezone

from ..forms import QuestionForm
from ..models import Question, Board
from common.models import PointTokenTransaction
import time
import datetime  # Correctly import the datetime module
# Remove: from datetime import datetime, timezone to avoid conflicts
from background_task import background
from django.urls import include, path
import logging

import ccxt
import pandas as pd
import zoneinfo
from decimal import Decimal

@background(schedule=60)
def verify_prediction_task(user_id, question_id, duration, crypto, direction, price_change):
    print('+'*100)
    print("undergoing verification")
    print('+'*100)
    points_map = {
        '1 hour': 1,
        '2 hours': 2,
        '4 hours': 3,
        '6 hours': 5,
        '8 hours': 8,
        '12 hours': 10,
        '24 hours': 20,
        'one week': 40,
        'one month': 80,
        'one year': 160
    }
    User = get_user_model()
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        logging.error(f"유저 아이디 {user_id}가 존재하지 않습니다!")

    try:
        question = Question.objects.get(id=question_id)
    except Question.DoesNotExist:
        return
    bitget = ccxt.bitget()
    symbol = "BTC/USDT:USDT"
    if crypto == "BTC":
        symbol = "BTC/USDT:USDT"
    elif crypto == "ETH":
        symbol = "ETH/USDT:USDT"
    ohlcv = bitget.fetch_ohlcv(symbol, "1h")
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    current_price = df["close"].values[-1]
    actual_price_change = ((Decimal(current_price) - Decimal(question.initial_price)) / Decimal(question.initial_price)) * 100
    correct_prediction = (actual_price_change > 0 and direction == "bullish") or \
                         (actual_price_change < 0 and direction == "bearish")

    if correct_prediction:
        base_point = points_map[duration]
        accuracy = abs(Decimal(actual_price_change) - Decimal(price_change))
        if accuracy <= 1:
            multiplier = 2
        elif accuracy <= 3:
            multiplier = 1.5
        else:
            multiplier = 1
        awarded_points = base_point * multiplier

        # award tokens to user
        user.profile.tokens += awarded_points
        user.save()
        PointTokenTransaction.objects.create(
            user = user,
            points = 0,
            tokens = awarded_points,
            reason = "예측 정확도에 따른 토큰 지급"
        )
        logging.info(f"축하합니다! 예측이 정확했고 {awarded_points} 토큰을 얻었습니다.")
    else:
        # no tokens awarded to user
        PointTokenTransaction.objects.create(
            user=user,
            points=0,
            tokens=0,
            reason="예측 정확도에 따른 토큰 지급"
        )
        logging.info("예측이 틀렸네요. 다음에 또 도전해 보세요!")

@login_required(login_url="common:login")
def question_create(request, board_name=None):
    if request.method == "POST":
        user_id = request.user.id
        form = QuestionForm(request.POST)
        if form.is_valid():
            subject = form.cleaned_data.get("subject")
            content = form.cleaned_data.get("content")
            question = form.save(commit=False)
            question.author = request.user
            if board_name:
                board = get_object_or_404(Board, name=board_name)
                question.board = board
            question.create_date = timezone.now()

            if board_name == "perceptive":
                word_count = len(content.split())
                if word_count < 55:
                    messages.error(request, "내용은 최소 55단어 이상이어야 합니다!")
                    request.session["stored_form_data"] = request.POST
                    return redirect('pybo:question_create', board_name=board_name)
                crypto = request.POST.get("crypto")
                duration = request.POST.get("duration")
                direction = request.POST.get("direction")
                price_change = request.POST.get("price_change")

                # fetch initial price
                bitget = ccxt.bitget()
                symbol = "BTC/USDT:USDT"
                if crypto == "BTC":
                    symbol = "BTC/USDT:USDT"
                elif crypto == "ETH":
                    symbol = "ETH/USDT:USDT"
                ohlcv = bitget.fetch_ohlcv(symbol, "1h")
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                initial_price = df.iloc[-1, 4]
                question.initial_price = initial_price

                # Add validation for price_change based on direction
                try:
                    price_change_value = float(price_change.strip('%'))
                except ValueError:
                    messages.error(request, "올바르지 않은 가격 변동 값입니다!")
                    request.session["stored_form_data"] = request.POST
                    return redirect('pybo:question_create', board_name=board_name)
                if direction == "bullish" and price_change_value < 0:
                    messages.error(request, "상승 추세를 위한 가격 변동은 0% 이상이어야 합니다!")
                    request.session["stored_form_data"] = request.POST
                    return redirect('pybo:question_create', board_name=board_name)
                elif direction == "bearish" and price_change_value > 0:
                    messages.error(request, "하락 추세를 위한 가격 변동은 0% 이하이어야 합니다!")
                    request.session["stored_form_data"] = request.POST
                    return redirect('pybo:question_create', board_name=board_name)

                last_prediction = Question.objects.filter(
                    author=request.user,
                    subject__regex=rf"\[{crypto}\]\[{duration}\].*"
                ).order_by('-create_date').first()

                duration_hours = {
                    '1 hour': 1,
                    '2 hours': 2,
                    '4 hours': 4,
                    '6 hours': 6,
                    '8 hours': 8,
                    '12 hours': 12,
                    '24 hours': 24,
                    'one week': 24 * 7,
                    'one month': 24 * 30,
                    'one year': 24 * 365
                }
                if last_prediction:
                    end_time = last_prediction.create_date + datetime.timedelta(
                        hours=duration_hours[duration])
                    if timezone.now() < end_time:
                        messages.error(request,
                                       '해당 코인에 대해서 이 기간 동안 이전 예측이 끝나기 전까지 다른 예측을 게시할 수 없습니다!')
                        request.session["stored_form_data"] = request.POST
                        return redirect('pybo:question_create', board_name=board_name)
                    else:
                        subject = f"[{crypto}][{duration}][{direction}][{price_change}] {subject}"
                        question.subject = subject
                        question.content = content
                else:
                    subject = f"[{crypto}][{duration}][{direction}][{price_change}] {subject}"
                    question.subject = subject
                    question.content = content
                question.save()

                verify_prediction_task(
                    user_id = user_id,
                    question_id = question.id,
                    duration = duration,
                    crypto = crypto,
                    direction = direction,
                    price_change = float(price_change.strip("%")),
                    schedule = datetime.timedelta(hours=duration_hours[duration])
                )
                messages.success(request, "당신의 예측이 기록되었으며, 시간이 되면 검증이 진행될겁니다!")

            elif board_name == "technical_blog" or board_name == "trading_blog":
                word_count = len(content.split())
                if word_count < 55:
                    messages.error(request, "내용은 최소 55단어 이상이어야 합니다!")
                    request.session["stored_form_data"] = request.POST
                    return redirect('pybo:question_create', board_name=board_name)
                question.save()
            # question.save()
            # Assign points based on the board
            if board_name == "perceptive":
                points_for_question = 5
            elif board_name in ["free_board", "technical_blog", "trading_blog"]:
                points_for_question = 3
            else:
                points_for_question = 0  # Default points for other boards if any
            # Create a PointTokenTransaction after saving the question
            eng_to_kor_name = {'perceptive': '관점공유 게시판', 'free_board': '자유게시판', 'technical_blog': '기술 블로그',
                               'trading_blog': '트레이딩 블로그'}
            kor_board_name = eng_to_kor_name[board_name]
            PointTokenTransaction.objects.create(
                user=request.user,
                points=points_for_question,
                tokens=0,  # Assuming you only want to deal with points
                reason=f"{kor_board_name}에 포스팅"
            )

            if 'stored_form_data' in request.session:
                del request.session['stored_form_data']

            if board_name:
                return redirect('pybo:board_view', board_name=board_name)
            else:
                return redirect("pybo:index")
        else:
            # store the form data in the session in case of form validation failure
            request.session["stored_form_data"] = request.POST
            return redirect('pybo:question_create', board_name=board_name)
    else:
        if 'stored_form_data' in request.session:
            form = QuestionForm(request.session['stored_form_data'])
            del request.session['stored_form_data']
        else:
            form = QuestionForm()
    context = {"form": form}
    if board_name == "perceptive":
        context["is_perceptive_board"] = True
    return render(request, "pybo/question_form.html", context)


@login_required(login_url="common:login")
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
            content = form.cleaned_data.get("content")
            if question.board.name == "perceptive" or question.board.name == "technical_blog" or question.board.name == "trading_blog":
                # Validate word count for content
                word_count = len(content.split())
                if word_count < 55:
                    messages.error(request, "내용은 최소 55단어 이상이어야 합니다!")
                    request.session["stored_form_data"] = request.POST
                    return redirect('pybo:question_modify', question_id=question.id)
            question.save()
            if 'stored_form_data' in request.session:
                del request.session['stored_form_data']
            return redirect('pybo:detail', question_id=question.id)
        else:
            request.session["stored_form_data"] = request.POST
            return redirect('pybo:question_modify', question_id=question.id)
    else:
        if 'stored_form_data' in request.session:
            form = QuestionForm(request.session["stored_form_data"])
            del request.session["stored_form_data"]
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
    if question.board.name == "perceptive":
        points_for_question = -5
    elif question.board.name in ["free_board", "technical_blog", "trading_blog"]:
        points_for_question = -3
    else:
        points_for_question = 0  # Default points for other boards if any
    # Create a PointTokenTransaction after saving the question
    eng_to_kor_name = {'perceptive': '관점공유 게시판', 'free_board': '자유게시판', 'technical_blog': '기술 블로그',
                       'trading_blog': '트레이딩 블로그'}
    kor_board_name = eng_to_kor_name[question.board.name]
    PointTokenTransaction.objects.create(
        user=request.user,
        points=points_for_question,
        tokens=0,  # Assuming you only want to deal with points
        reason=f"{kor_board_name}에 포스팅 삭제"
    )
    return redirect('pybo:index')
