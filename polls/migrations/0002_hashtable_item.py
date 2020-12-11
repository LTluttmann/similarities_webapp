# Generated by Django 3.1.4 on 2020-12-09 20:03

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('image', models.BinaryField()),
            ],
        ),
        migrations.CreateModel(
            name='HashTable',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('hash_key', models.CharField(max_length=64)),
                ('item_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='polls.item')),
            ],
        ),
    ]
