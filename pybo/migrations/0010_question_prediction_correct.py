# Generated by Django 4.2.5 on 2023-11-14 06:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pybo', '0009_question_initial_price'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='prediction_correct',
            field=models.BooleanField(default=False),
        ),
    ]