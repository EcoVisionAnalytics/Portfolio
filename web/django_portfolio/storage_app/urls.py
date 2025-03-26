from django.urls import path
from .views import get_client_folders, get_client_projects, get_project_files, get_user_files

urlpatterns = [
    path('clients/', get_client_folders, name='client-folders'),  # ✅ Fix: Add this
    path('clients/<str:client_folder>/projects/', get_client_projects, name='client-projects'),
    path('clients/<str:client_folder>/<str:project_name>/files/', get_project_files, name='project-files'),
    path('files/', get_user_files, name='user-files'), 
]

