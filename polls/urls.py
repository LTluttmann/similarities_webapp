from django.urls import path

from . import views


app_name = "polls"
urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    # ex: /polls/5/vote/
    path('1/vote/', views.vote, name='vote'),
]