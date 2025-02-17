import logging

def handle_whatsapp_request(form_data):
    """
    A placeholder for a WhatsApp chatbot.
    Typically, you'd parse the incoming message from form_data["Body"].
    Then respond with forecast or a quick text.
    """
    user_msg = form_data.get("Body", "").strip().lower()
    # A simple rule-based approach:
    if "price" in user_msg:
        # Return a fixed reply or call your forecast code
        return "Today's Apple Price is ₹45.7. Next 7 days forecast: ~45 - 47 range."
    elif "hi" in user_msg or "hello" in user_msg:
        return "Hello from Market Intel Bot! Type 'price' to get apple rates."
    else:
        return "Sorry, I didn't understand. Type 'price' for updates."
