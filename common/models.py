from django.db import models, IntegrityError, transaction
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from pybo.models import Question, Answer, Comment
import secrets

class Attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="attendances")
    timestamp = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ["-timestamp"]

# Create your models here.
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    intro = models.CharField(max_length=300, blank=True, null=True, verbose_name="한줄소개")
    image = models.ImageField(upload_to="profile_images/", null=True, blank=True)
    score = models.PositiveIntegerField(default=0)
    tokens = models.PositiveIntegerField(default=0)
    referral_code = models.CharField(max_length=8, unique=True, null=True, blank=True)
    referred_by = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name="referrals")

    def generate_unique_referral_code(self):
        referral_code = secrets.token_urlsafe(8)[:8]
        while Profile.objects.filter(referral_code=referral_code).exists():
            referral_code = secrets.token_urlsafe(8)[:8]
        return referral_code

    def save(self, *args, **kwargs):
        if not self.referral_code:
            self.referral_code = self.generate_unique_referral_code()
            # Attempt to save with the new referral code in a transaction
            while True:
                try:
                    with transaction.atomic():
                        super(Profile, self).save(*args, **kwargs)
                    break  # If save was successful, break out of the loop
                except IntegrityError:
                    # If an IntegrityError occurred, it could be due to a referral code collision.
                    # Generate a new code and try again.
                    self.referral_code = self.generate_unique_referral_code()
        else:
            # If the referral code is already set, just save the instance normally.
            super(Profile, self).save(*args, **kwargs)

    def calculate_score(self):
        score = 0
        user = self.user

        # Add logic to calculate the score based on past posts, answers, comments, and likes.
        for question in Question.objects.filter(author=user):
            if question.board.name == "perceptive":
                score += 5
            elif question.board.name in ["free_board", "technical_blog", "trading_blog"]:
                score += 3

            # Likes received for questions
            score += 2 * question.voter.count()

        for answer in Answer.objects.filter(author=user):
            if answer.question.board.name == "perceptive":
                score += 4
            elif answer.question.board.name in ["free_board", "technical_blog", "trading_blog"]:
                score += 2

            # Likes received for answers
            score += 2 * answer.voter.count()

        for comment in Comment.objects.filter(author=user):
            score += 1

        # Likes given
        score += user.voter_question.count()  # Updated line
        score += user.voter_answer.count()  # Updated line
        return score

    def __str__(self):
        return self.user.username

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        profile = Profile.objects.create(user=instance)
    else:
        profile = instance.profile

    profile.score = profile.calculate_score()
    profile.save()

class PointTokenTransaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="transactions")
    points = models.IntegerField(default=0)
    tokens = models.IntegerField(default=0)
    reason = models.CharField(max_length=500)
    timestamp = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ["-timestamp"]
    def __str__(self):
        return f"{self.user.username} - 포인트: {self.points}, 토큰: {self.tokens}, 내역: {self.reason}"