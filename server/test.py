import requests

BASE_URL = "http://127.0.0.1:8000"

# -----------------------
# 1. Signup
# -----------------------
signup_payload = {
    "username": "michael",
    "full_name": "Mike jacson",
    "email": "jacdkson@example.com",
    "password": "StrongPassword123!"
}

signup_resp = requests.post(f"{BASE_URL}/users/signup-email", json=signup_payload)
print("Signup:", signup_resp.status_code, signup_resp.json())

# -----------------------
# 2. Login
# -----------------------
login_payload = {
    "username": "michael",
    "password": "StrongPassword123!"
}

login_resp = requests.post(f"{BASE_URL}/users/login-email", json=login_payload)
print("Login:", login_resp.status_code, login_resp.json())

tokens = login_resp.json().get("data") or {}
access_token = tokens.get("access_token")
refresh_token = tokens.get("refresh_token")
headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

# -----------------------
# 3. Upload a screenshot
# -----------------------
if access_token:
  # Upload test
    with open("screenshot.png", "rb") as f:
        files = {"file": f}
        upload_resp = requests.post(f"{BASE_URL}/uploads/", headers=headers, files=files)

    print("Upload status:", upload_resp.status_code)
    try:
        print("Upload response:", upload_resp.json())
    except Exception:
        print("Upload raw response:", upload_resp.text)

# -----------------------
# 4. Get insights
# -----------------------
if access_token:
    insights_resp = requests.get(f"{BASE_URL}/insights/", headers=headers)
    print("Insights:", insights_resp.status_code, insights_resp.json())

# -----------------------
# 5. Logout
# -----------------------
if access_token and refresh_token:
    logout_payload = {"refresh_token": refresh_token}
    logout_resp = requests.post(f"{BASE_URL}/users/logout", headers=headers, json=logout_payload)
    print("Logout:", logout_resp.status_code, logout_resp.json())
