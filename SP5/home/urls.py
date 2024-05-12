from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('first/',views.first),
    path('about/',views.about),
    path('first/random',views.random),
    path('first/logistic',views.logistic),
    path('first/knn',views.knn),
    path('first/tree',views.tree),
    path('first/result',views.result),
    path('first/visual',views.visual),
    # path('visual/result',views.result),
    path('first/visual2',views.visual2),
    path('first/visual3',views.visual3),
    path('first/visual4',views.visual4),
    path('first/accuracy',views.accuracy_metrics),

]