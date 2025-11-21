from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Chat, Category
from django.http import JsonResponse
import requests
from django.conf import settings
from django.db.models import Q

FASTAPI_BASE = "http://localhost:5555/DevTune/chat"

def home_view(request):
    """
    Home view that auto-creates a temporary chat session for logged-in users
    """
    categories = Category.objects.all()
    user_chats = []
    current_chat = None
    
    if request.user.is_authenticated:
        # Get all user chats (excluding temporary ones)
        user_chats = Chat.objects.filter(
            Chat_owner=request.user,
            Chat_is_temporary=False
        ).order_by('-Chat_createdat')
        
        # Check if there's an active temporary chat
        temp_chat = Chat.objects.filter(
            Chat_owner=request.user,
            Chat_is_temporary=True
        ).first()
        
        if temp_chat:
            current_chat = temp_chat
        else:
            # Only create a new temp chat if there are no permanent chats OR explicitly requested
            # This prevents auto-creation when user just wants to browse their chats
            should_create_temp = request.GET.get('new') == '1' or user_chats.count() == 0
            
            if should_create_temp:
                # Create a new temporary chat with default category
                default_category = Category.objects.filter(Category_is_default=True).first()
                if not default_category:
                    default_category = Category.objects.first()
                
                utility_params = {"completion_type": "main"}
                if default_category and default_category.Category_name == "QA_Normal":
                    utility_params["completion_type"] = "main"
                
                current_chat = Chat.objects.create(
                    Chat_owner=request.user,
                    Chat_category=default_category,
                    Chat_Active=True,
                    Chat_is_temporary=True,
                    Chat_utility_params=utility_params
                )
            elif user_chats.exists():
                # If user has chats and no temp chat, show the most recent one
                current_chat = user_chats.first()
    
    return render(request, 'devtune/devtune_home.html', {
        "categories": categories,
        "user_chats": user_chats,
        "current_chat": current_chat,
    })


@login_required
def create_chat(request, category_slug=None):
    """
    Create a new chat session (converts temp to permanent or creates new)
    """
    category = None
    utility_params = {}

    if category_slug:
        category = get_object_or_404(Category, Category_slug=category_slug)

        if category.Category_name == "QA_Normal":
            utility_params["completion_type"] = "main"
        elif category.Category_name == "Temp":
            utility_params["completion_type"] = "chat"
        elif category.Category_name == "Generator":
            utility_params["completion_type"] = "code_generation"
        elif category.Category_name == "Reviewer":
            utility_params["completion_type"] = "code_review"
        elif category.Category_name == "Summarizer":
            utility_params["completion_type"] = "summary"
        else:
            utility_params["completion_type"] = "chat"

    if request.method == "POST":
        # Delete any existing temporary chats
        Chat.objects.filter(
            Chat_owner=request.user,
            Chat_is_temporary=True
        ).delete()
        
        # Create new permanent chat
        chat = Chat.objects.create(
            Chat_owner=request.user,
            Chat_category=category,
            Chat_Active=True,
            Chat_is_temporary=False,
            Chat_utility_params=utility_params
        )
        return redirect("devtune:home_view")

    return render(request, "devtune/create_chat.html", {"category": category})


@login_required
def chat_panel(request, slug):
    """
    Display a specific chat (for old chats in sidebar)
    Redirects to home if viewing a temporary chat
    """
    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    
    # If this is a temporary chat, redirect to home view
    if chat.Chat_is_temporary:
        return redirect("devtune:home_view")
    
    user_chats = Chat.objects.filter(
        Chat_owner=request.user,
        Chat_is_temporary=False
    ).order_by('-Chat_createdat')
    
    return render(request, "devtune/chat_panel.html", {
        "chat": chat,
        "user_chats": user_chats,
        "current_chat": chat,
    })


@login_required
def send_message_ajax(request, slug):
    """
    AJAX endpoint that sends user message to FastAPI
    Converts temporary chat to permanent on first message
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    
    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    msg = request.POST.get("message", "").strip()
    
    if not msg:
        return JsonResponse({"error": "Empty message"}, status=400)
    
    # Track if this was a temporary chat
    was_temporary = chat.Chat_is_temporary
    
    # If this is the first message in a temporary chat, convert it to permanent
    if chat.Chat_is_temporary:
        chat.Chat_is_temporary = False
        # Generate a title from the first message (first 50 chars)
        chat.Chat_title = msg[:50] + ("..." if len(msg) > 50 else "")
        chat.Chat_utility_params["completion_type"] = "main"
        chat.save()
    
    payload = {
        "username": request.user.username,
        "session_id": chat.Chat_session_id,
        "prompt": msg,
        "chat_history": [],
        "utility_params": chat.Chat_utility_params
    }
    
    try:
        res = requests.post(f"{FASTAPI_BASE}/complete", json=payload, timeout=30)
        if res.status_code == 200:
            return JsonResponse({
                **res.json(),
                "chat_converted": was_temporary,
                "chat_title": chat.Chat_title,
                "chat_slug": chat.Chat_slug
            })
        else:
            return JsonResponse({"error": "FastAPI error"}, status=500)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)


@login_required
def delete_chat(request, slug):
    """
    Delete a chat session
    """
    if request.method == "POST":
        chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
        chat.delete()
        return JsonResponse({"success": True})
    return JsonResponse({"error": "Invalid request"}, status=400)


@login_required
def new_chat(request):
    """
    Create a new temporary chat and redirect to home
    """
    # Delete existing temporary chats
    Chat.objects.filter(
        Chat_owner=request.user,
        Chat_is_temporary=True
    ).delete()
    
    # Redirect to home with flag to create new temp chat
    return redirect("devtune:create_chat")