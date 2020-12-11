import datetime

from django.db import models
from django.utils import timezone


class Item(models.Model):
    id = models.IntegerField(primary_key=True)
    image = models.BinaryField(null=True)
    embedding = models.BinaryField(null=True)
    train = models.BooleanField(default=True)
    item_class = models.IntegerField(null=True)


class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


class Choice(models.Model):
    # query = models.ForeignKey(Item, on_delete=models.CASCADE)
    # neighbor = models.ForeignKey(Item, on_delete=models.CASCADE)
    # baseline = models.ForeignKey(Item, on_delete=models.CASCADE)
    choice = models.CharField(max_length=200, default=0)
