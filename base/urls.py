# myapp/urls.py
from django.urls import path
from django.urls import re_path as url  # Import the url function
from .views import index,index1,result, predictImage,HomePage,SignupPage,LoginPage,LogoutPage,segmentImage


urlpatterns = [
    path('',HomePage,name='home'),
    path('index/', index, name='index'),
    path('index1/', index1, name='index1'),
    path('index1/result/', result, name='result'),

    path('signup/',SignupPage,name='signup'),
    path('login/',LoginPage,name='login'),
    path('logout/',LogoutPage,name='logout'),
    url('^$',index,name='homepage'),

    # url('^$',index,name='homepage'),
    url('predictImage',predictImage,name='predictImage'),
    path('segment_image/', segmentImage, name='segment_image'),
]