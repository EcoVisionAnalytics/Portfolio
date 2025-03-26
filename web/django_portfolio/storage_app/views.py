from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)
bucket_name = "client-data-lake"
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_client_folders(request):
    """Staff see all client folders; regular users see only their assigned folder."""

    bucket_name = "client-data-lake"
    storage_client = storage.Client()

    try:
        if request.user.is_staff:
            # Staff can see all top-level client folders
            blobs = storage_client.list_blobs(bucket_name, delimiter="/")
            client_folders = {blob.name.split("/")[0] for blob in blobs if "/" in blob.name}
        else:
            # Regular users see only their assigned client folder
            client_folders = {request.user.client_folder} if request.user.client_folder else set()

        return Response({"client_folders": list(client_folders)})
    
    except GoogleAPIError as e:
        logger.error(f"Google Cloud API Error: {e}")
        return Response({"error": "Failed to connect to Google Cloud Storage. Please check your credentials."}, status=500)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return Response({"error": "An unexpected error occurred while retrieving client folders."}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_client_projects(request, client_folder=None):
    """List all subdirectories (projects) inside a client's folder."""

    storage_client = storage.Client()
    bucket_name = "client-data-lake"

    # If user is not staff, enforce access to their own client folder
    if not request.user.is_staff:
        client_folder = request.user.client_folder

    if not client_folder:
        return Response({"error": "No client folder assigned or selected."}, status=403)

    blobs = storage_client.list_blobs(bucket_name, prefix=f"{client_folder}/", delimiter="/")
    projects = {blob.name.split("/")[1] for blob in blobs if len(blob.name.split("/")) > 1}

    return Response({"projects": list(projects)})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_project_files(request, client_folder, project_name):
    """List all files inside a project. Staff users can access any client folder."""

    storage_client = storage.Client()
    bucket_name = "client-data-lake"

    # If the user is not staff, they can only access their assigned client folder
    if not request.user.is_staff and request.user.client_folder != client_folder:
        return Response({"error": "Unauthorized"}, status=403)

    project_path = f"{client_folder}/{project_name}/"
    blobs = storage_client.list_blobs(bucket_name, prefix=project_path)

    files = [{"name": blob.name.split("/")[-1], "url": f"https://storage.googleapis.com/{bucket_name}/{blob.name}"} for blob in blobs]

    return Response({"files": files})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_files(request):
    """Return a plain list of files the user has access to."""
    
    bucket_name = "client-data-lake"
    storage_client = storage.Client()

    try:
        if request.user.is_staff:
            # Staff members see ALL files in the bucket
            blobs = storage_client.list_blobs(bucket_name)
        else:
            # Regular users only see files inside their assigned client folder
            if not request.user.client_folder:
                return Response({"error": "No client folder assigned."}, status=403)
            blobs = storage_client.list_blobs(bucket_name, prefix=f"{request.user.client_folder}/")

        files = [
            {"name": blob.name, "url": f"https://storage.googleapis.com/{bucket_name}/{blob.name}"}
            for blob in blobs
        ]
        
        return Response({"files": files})

    except GoogleAPIError as e:
        logger.error(f"Google Cloud API Error: {e}")
        return Response({"error": "Failed to connect to Google Cloud Storage. Please check your credentials."}, status=500)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return Response({"error": "An unexpected error occurred while retrieving files."}, status=500)