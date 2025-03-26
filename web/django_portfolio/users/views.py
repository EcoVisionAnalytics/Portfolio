from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.middleware.csrf import get_token

def get_csrf_token(request):
    """Returns CSRF token for React to use."""
    return JsonResponse({"csrfToken": get_token(request)})

@login_required
def get_user_data(request):
    """Returns user data for authenticated requests."""
    return JsonResponse({
        "username": request.user.username,
        "is_staff": request.user.is_staff
    })


