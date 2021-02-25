from django.db import models


class Item(models.Model):
    id = models.IntegerField(primary_key=True)
    train = models.BooleanField(default=True)
    item_gender = models.CharField(max_length=5)
    item_type = models.CharField(max_length=32)
    item_color = models.CharField(max_length=16)


class Image(models.Model):
    item = models.ForeignKey(Item, on_delete=models.CASCADE)
    image = models.BinaryField(null=True)


class Choice(models.Model):
    # query = models.ForeignKey(Item, on_delete=models.CASCADE)
    # neighbor = models.ForeignKey(Item, on_delete=models.CASCADE)
    # baseline = models.ForeignKey(Item, on_delete=models.CASCADE)
    choice = models.CharField(max_length=200, default=0)
