from django.db import models
from django.conf import settings


class UserFile(models.Model):
    """Stores metadata for user-specific files in GCS."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='storage_files'  # Unique related name
    )
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=1024)  # GCS path
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.file_name}"
