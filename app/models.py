from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Waitlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    twitter = db.Column(db.String(80), unique=True,nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    wallet = db.Column(db.String(120), nullable=True)
    verify_code = db.Column(db.String(6))
    referral_code = db.Column(db.String(20), unique=True, nullable=True)
    referred_by = db.Column(db.String(20), nullable=True)
    verified = db.Column(db.Boolean, default=False)
    login_token = db.Column(db.String(32), nullable=True)
    token_expires = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)