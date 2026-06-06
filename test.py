import requests

TOKEN = "8659715066:AAEJiPKZHJ9kFX3MfH3JanWJ21ZKzCGYqyQ"
CHAT_ID = "8060242037"

requests.post(
    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    data={
        "chat_id": CHAT_ID,
        "text": "Test YOLO 🚀"
    }
)