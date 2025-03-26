from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings

class CustomUser(AbstractUser):
    """Custom user model extending Django's AbstractUser"""
    first_login = models.BooleanField(default=True)
    organization = models.CharField(max_length=255, blank=True, null=True)
    client_folder = models.CharField(max_length=155, null=True, blank=True)
    
    def __str__(self):
        return self.username 
    
class UserFile(models.Model):
    """Stores metadata for user-specific files in GCS."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="user_files"  # Prevents reverse accessor conflict
    )
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=1024, default='default_path')  # GCS path
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file_name} - {self.user.username}"
