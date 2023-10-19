from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class Board(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    def __str__(self):
        return self.name

class Question(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question')
    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name='freeboard_questions', null=True)
    subject = models.CharField(max_length=200)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    voter = models.ManyToManyField(User, related_name='voter_question') # add voter
    def __str__(self):
        return self.subject
    def save(self, *args, **kwargs):
        is_new = self.pk is None  # Check if the question is new
        super().save(*args, **kwargs)  # Call the super class's save method
        if is_new:  # If the question is new, then update the score
            self.author.profile.score += 5 if self.board.name == "perceptive" else 3
            self.author.profile.save()
    def delete(self, *args, **kwargs):
        if self.board.name == "perceptive":
            self.author.profile.score -= 5
        else:
            self.author.profile.score -= 3
        self.author.profile.save()
        super().delete(*args, **kwargs)

class Answer(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_answer')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    voter = models.ManyToManyField(User, related_name='voter_answer')
    def save(self, *args, **kwargs):
        is_new = self.pk is None
        super().save(*args, **kwargs)
        if is_new:
            if self.question.board.name == "perceptive":
                self.author.profile.score += 4
            else:
                self.author.profile.score += 2
            self.author.profile.save()

    def delete(self, *args, **kwargs):
        if self.question.board.name == "perceptive":
            self.author.profile.score -= 4
        else:
            self.author.profile.score -= 2
        self.author.profile.save()
        super().delete(*args, **kwargs)

class Comment(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    question = models.ForeignKey(Question, null=True, blank=True, on_delete=models.CASCADE)
    answer = models.ForeignKey(Answer, null=True, blank=True, on_delete=models.CASCADE)
    def save(self, *args, **kwargs):
        is_new = self.pk is None
        super().save(*args, **kwargs)
        if is_new:
            self.author.profile.score += 1
            self.author.profile.save()

    def delete(self, *args, **kwargs):
        self.author.profile.score -= 1
        self.author.profile.save()
        super().delete(*args, **kwargs)
