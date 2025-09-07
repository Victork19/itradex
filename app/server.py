from flask import Flask, abort,render_template, request, redirect, url_for, session, jsonify, flash
from datetime import datetime, timedelta
import random, string, jwt, time, secrets
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from models import db, Waitlist, Admin
import os
from flask_limiter import Limiter 
from flask_limiter.util import get_remote_address
import requests
from email_sender import send_signup_code,  is_valid_email 

app = Flask(__name__)
app.config.from_object(Config)
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=0.00030)

limiter = Limiter(
    get_remote_address,
    app=app,
)

db.init_app(app)

with app.app_context():
    db.create_all()




verification_codes = {}

fingerprint_log = {}

def get_fingerprint():
  
    ip = request.remote_addr or "unknown"
    ua = request.headers.get("User-Agent", "unknown")

    # Optional: check a fingerprint cookie
    token = request.cookies.get("fp_token", "")
    return f"{ip}:{ua}:{token}"

def is_suspicious(fingerprint):
    now = time.time()
    window = 60  # seconds
    max_requests = 50

    if fingerprint not in fingerprint_log:
        fingerprint_log[fingerprint] = []

    # keep only recent requests
    fingerprint_log[fingerprint] = [t for t in fingerprint_log[fingerprint] if now - t < window]

    # add this request
    fingerprint_log[fingerprint].append(now)

    # too many hits in short time → bot
    if len(fingerprint_log[fingerprint]) > max_requests:
        return True

    return False

@app.before_request
def block_bots():
    fingerprint = get_fingerprint()

    # quick UA filter (optional)
    bad_uas = ["curl", "python-requests", "wget"]
    ua = request.headers.get("User-Agent", "").lower()
    if any(bad in ua for bad in bad_uas):
        abort(403)  # forbidden

    # rate / fingerprint check
    if is_suspicious(fingerprint):
        abort(429)  # too many requests


def generate_referral_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("leaderboard"))

  


    referred_by = session.get('referred_by', '')

    return render_template("waitlist.html", referred_by=referred_by)

@app.route("/login")
def login_page():
    # Check session first
    if "user" in session:
        return redirect(url_for("leaderboard"))

    # Check persistent login token
    token = request.cookies.get("login_token")
    if token:
        # Validate token against DB
        user = Waitlist.query.filter_by(login_token=token).first()
        if user and user.token_expires > datetime.utcnow():
            # Optionally refresh session
            session["user"] = user.email
            return redirect(url_for("leaderboard"))

        # Token exists but expired → allow access to login page
        return render_template("login.html")

    # No session, no token → deny access (redirect home or show 403)
    return redirect(url_for("home"))



@app.route('/r/<ref_code>')
def referral(ref_code):
    # Store the referral code in the session
    session['referred_by'] = ref_code
    print(f'Ref; {ref_code}')
    # Redirect to the main waitlist page
    return redirect(url_for('home'))


@app.route('/submit_waitlist', methods=['POST'])
@limiter.limit("5 per minute")
def submit_waitlist():
    data = request.get_json()
    twitter = data.get('username')
    email = data.get('email')
    wallet = data.get('wallet', '')
    honeypot = request.form.get("extra_field")
    referred_by = data.get('referred_by')
   

    # reCAPTCHA token
    token = request.form.get("g-recaptcha-response")
    token = data.get("g-recaptcha-response")
    if not token:
        return jsonify({"success": False, "error": "Captcha token missing"}), 400

    # Verify reCAPTCHA v2
    secret_key = app.config['RECAPTCHA_SECRET_KEY']
    resp = requests.post(
        "https://www.google.com/recaptcha/api/siteverify",
        data={"secret": secret_key, "response": token}
    )
    result = resp.json()
    if not result.get("success"):
        return jsonify({"success": False, "error": "Captcha verification failed"}), 400


  


    # Optional: remove token to prevent reuse
    # session.pop("csrf_token", None)
    # session.pop("form_loaded_at", None)


    if honeypot: 
        return "Bot detected", 403
    

    # Validate required fields
    if not twitter or not email:
        return jsonify({'success': False, 'message': 'Username and email are required'}), 400
    if not is_valid_email(email):
        return jsonify({'success': False, 'message': 'Invalid email address'}), 400

    if Waitlist.query.filter_by(twitter=twitter).first():
        return jsonify({'success': False, 'error': 'Twitter username is already taken'}), 400
    
    # Check if email already exists
    if Waitlist.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered'}), 400

    # Generate unique referral code
    referral_code = generate_referral_code()
    while Waitlist.query.filter_by(referral_code=referral_code).first():
        referral_code = generate_referral_code()  # ensure uniqueness
    
    import secrets, string
    login_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
    token_expires = datetime.utcnow() + timedelta(days=30)


    # Create waitlist entry
    entry = Waitlist(
        twitter=twitter,
        email=email,
        wallet=wallet,
        referral_code=referral_code,
        referred_by=referred_by,
        verified=False,
        login_token=login_token,
        token_expires=token_expires
    )
    db.session.add(entry)
    db.session.commit()

    
    # Send email
    send_signup_code(email=email, code=login_token)
    # print(f"Verification code for {email}: {login_token}") 
    return jsonify({'success': True, 'message': 'Verification code sent'})


@app.route('/resend_code', methods=['POST'])
def resend_code():
    data = request.get_json()
    email = data.get("email", "").strip().lower()

    if not email:
        return jsonify({"success": False, "error": "Email is required"}), 400

    # Check if email exists in DB
    user = Waitlist.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "error": "Email not found"}), 404

    # Use existing code if present, otherwise generate new persistent alphanumeric code
    if user.login_token and user.token_expires > datetime.utcnow():
        code = user.login_token
    else:
        import secrets, string
        code = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        user.login_token = code
        user.token_expires = datetime.utcnow() + timedelta(days=30)  # persistent session
        db.session.commit()

    try:
    
        send_signup_code(email=email, code=code)
        # print(f"Verification code for {email}: {code}") 
    except Exception as e:
        print(f"Email error: {e}")
        return jsonify({"success": False, "error": "Failed to send code"}), 500

    return jsonify({"success": True, "message": "Code sent"})


@app.route('/check_username', methods=['POST'])
def check_username():
    data = request.get_json()
    username = data.get('username')
    exists = Waitlist.query.filter_by(twitter=username).first() is not None
    return jsonify({'available': not exists})


@app.route('/check_referral', methods=['POST'])
def check_referral():
    data = request.get_json()
    code = data.get("code", "").strip()

    if not code:
        return jsonify({"valid": False, "error": "No referral code provided"})

    ref_entry = Waitlist.query.filter_by(referral_code=code).first()
    if ref_entry:
        return jsonify({"valid": True})
    else:
        return jsonify({"valid": False, "error": "Invalid referral code"})


@app.route("/verify_code", methods=["POST"])
def verify_code():
    data = request.json
    email = data.get("email")
    code = data.get("code")

    user = Waitlist.query.filter_by(email=email).first()
    if not user or user.login_token != code or user.token_expires < datetime.utcnow():
        return jsonify({"success": False, "error": "Invalid or expired code"}), 400

    session["user"] = email
    resp = jsonify({"success": True})
    resp.set_cookie("login_token", user.login_token, httponly=True, max_age=30*24*60*60)
    return resp

@app.route("/check_token", methods=["POST"])
def check_token():
    token = request.json.get("token")
    if not token:
        return jsonify({"valid": False})

    user = Waitlist.query.filter_by(login_token=token).first()
    if user and user.token_expires > datetime.utcnow():
        session["user"] = user.email  # refresh session
        return jsonify({"valid": True})
    return jsonify({"valid": False})


@app.route("/logout")
def logout():
    token = request.cookies.get("login_token")
    if token:
        user = Waitlist.query.filter_by(login_token=token).first()
        if user:
            user.login_token = None
            user.token_expires = None
            db.session.commit()

    session.clear()
    resp = redirect(url_for("home"))
    resp.delete_cookie("login_token")
    return resp


@app.route("/leaderboard")
def leaderboard():
    session_email = session.get("user")
    if not session_email:
        return redirect(url_for("login_page"))
    session_email = session_email.strip().lower()

    user = Waitlist.query.filter_by(email=session_email).first()
    if not user:
        return redirect(url_for("login_page"))

    # ===== Fake Scores ===== #
    fake_scores = [
        {"username": "****lf", "referrals": 1},
        {"username": "****utl", "referrals": 0},
        {"username": "****nax", "referrals": 0},
        {"username": "****rdar", "referrals": 0},
        {"username": "****03x", "referrals": 0},
        {"username": "****eon", "referrals": 1},
        {"username": "****rp", "referrals": 0},
    ]

    # ===== Real Users ===== #
    real_users = Waitlist.query.all()  # include all users
    real_scores = []
    for u in real_users:
        count = Waitlist.query.filter_by(referred_by=u.referral_code).count()
        masked_username = "****" + u.twitter[4:] if len(u.twitter) > 4 else "****" + u.twitter
        real_scores.append({
            "username": masked_username,
            "email": u.email.strip().lower(),
            "referrals": count
        })

    # ===== Merge & Sort All Scores ===== #
    all_scores = fake_scores + real_scores
    all_scores_sorted = sorted(all_scores, key=lambda x: x["referrals"], reverse=True)

    # ===== Current user position ===== #
    user_index = next(
        (i for i, s in enumerate(all_scores_sorted) if s.get("email") == session_email),
        None
    )
    position = user_index + 1 if user_index is not None else None

    # ===== Top 10 for display ===== #
    top_scores = all_scores_sorted[:10]

    # ===== If user is outside top 10, append them for visibility ===== #
    if position and position > 10:
        top_scores.append(all_scores_sorted[user_index])

    # ===== User referral count ===== #
    user_referral_count = Waitlist.query.filter_by(referred_by=user.referral_code).count()

    return render_template(
        "leaderboard.html",
        scores=top_scores,
        position=position,
        user=user,
        user_referral_count=user_referral_count
    )


# --- add near the top with your other globals ---
follow_clicks = {}  # map keys -> timestamp; key = username.lower() if provided, else remote_addr

# --- new endpoint: record that someone clicked Follow ---
@app.route('/record_follow', methods=['POST'])
def record_follow():
    data = request.get_json() or {}
    username = data.get('username')
    key = None
    if username:
        key = username.strip().lower()
    else:
        key = request.remote_addr
    follow_clicks[key] = time.time()
    return jsonify({"success": True})

# --- new endpoint: check whether a follow was recorded for username or IP ---
@app.route('/check_follow', methods=['POST'])
def check_follow():
    data = request.get_json() or {}
    username = data.get('username')
    # prefer exact username match
    if not username:
        if not username.strip().lower() in follow_clicks:
            return jsonify({"followed": True})
    # fallback to IP-based record
    if not request.remote_addr in follow_clicks:
        return jsonify({"followed": True})
    # also allow cookie 'follow_clicked' as a soft signal (client sets it)
    if not request.cookies.get('follow_clicked') == '1':
        return jsonify({"followed": True})
    return jsonify({"followed": False})


@app.route("/admin")
def admin_panel():
    if not session.get("admin"):
        abort(403)

    total_users = Waitlist.query.count()
    verified_users = Waitlist.query.filter_by(verified=True).count()
    users = Waitlist.query.order_by(Waitlist.id.desc()).all()

    return render_template(
        "admin.html",
        total_users=total_users,
        verified_users=verified_users,
        users=users
    )

ADMIN_EMAIL = "ukovictor8@gmail.com"
# Generate this once and store the hash
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        admin = Admin.query.filter_by(email=email).first()
        if admin and admin.check_password(password):
            session["admin"] = True
            return redirect(url_for("admin_panel"))
        else:
            flash("Invalid email or password", "danger")

    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin/login")


if __name__ == "__main__":
    app.secret_key = Config.SECRET_KEY
    app.run(debug=True, host="0.0.0.0", port=8000)

