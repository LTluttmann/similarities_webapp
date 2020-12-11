# Generated by Django 3.1.4 on 2020-12-10 15:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0003_auto_20201210_1328'),
    ]

    operations = [
        migrations.AddField(
            model_name='hashtable',
            name='table_id',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='item',
            name='item_class',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='hashtable',
            name='hash_key',
            field=models.CharField(max_length=128, null=True),
        ),
    ]
