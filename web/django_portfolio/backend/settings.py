from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Google Cloud Storage Configuration
DEFAULT_FILE_STORAGE = 'storages.backends.gcloud.GoogleCloudStorage'
GS_BUCKET_NAME = 'client-data-lake'
GS_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
# Security
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-default-key") 
DEBUG = True
ALLOWED_HOSTS = []
STATIC_URL = '/static/'  # 
STATICFILES_DIRS = [BASE_DIR / "static"]  
STATIC_ROOT = BASE_DIR / "staticfiles"
# Authentication & Redirects
AUTH_USER_MODEL = 'users.CustomUser'
LOGIN_URL = "/auth/login/"
LOGIN_REDIRECT_URL = "http://localhost:3000/"
LOGOUT_REDIRECT_URL = "/auth/login/"
ROOT_URLCONF = 'backend.urls' 
# CORS & CSRF Config
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]
CSRF_TRUSTED_ORIGINS = ["http://localhost:3000"]

# Installed Applications
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'storage_app',
    'users',
]
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
# Middleware Configuration
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Django REST Framework (API Security & Authentication)
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        "rest_framework.authentication.BasicAuthentication",
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
}
