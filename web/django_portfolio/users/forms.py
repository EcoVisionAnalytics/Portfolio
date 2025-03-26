from django import forms
from django.contrib.auth.forms import PasswordChangeForm

class CustomPasswordChangeForm(PasswordChangeForm):
    """Custom form to remove the extra verification text."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
        self.fields['new_password2'].help_text = ''

