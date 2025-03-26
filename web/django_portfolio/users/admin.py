from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['username', 'email', 'is_staff', 'is_active'] 
    

    fieldsets = UserAdmin.fieldsets + (
        ('Client Data Restrictions', {'fields': ('client_folder',)}),
    )

admin.site.register(CustomUser, CustomUserAdmin)
