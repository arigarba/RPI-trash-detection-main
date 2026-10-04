import requests

TOKEN = "8659715066:AAEJiPKZHJ9kFX3MfH3JanWJ21ZKzCGYqyQ"
CHAT_IDS = [
    "8060242037",
    "8416897843",
]

for chat_id in CHAT_IDS:
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": "Test YOLO 🚀"
        }
    )
    print(f"Chat ID {chat_id}: {r.json()}") # lihat error di sini