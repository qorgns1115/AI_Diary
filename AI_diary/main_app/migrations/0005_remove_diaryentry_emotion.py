# Generated by Django 5.1.3 on 2024-11-22 15:20

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0004_diary_image_path'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='diaryentry',
            name='emotion',
        ),
    ]