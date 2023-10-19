from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from pybo.models import Question, Answer, Comment

# Create your models here.
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    intro = models.CharField(max_length=300, blank=True, null=True, verbose_name="한줄소개")
    image = models.ImageField(upload_to="profile_images/", null=True, blank=True)
    score = models.PositiveIntegerField(default=0)

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