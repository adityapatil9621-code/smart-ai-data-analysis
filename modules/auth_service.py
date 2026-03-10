"""
Authentication service for login & registration
"""

import bcrypt
from auth_db import get_connection


# ===============================
# Password Hashing
# ===============================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ===============================
# Register User
# ===============================

def register_user(username, email, password):

    conn = get_connection()
    cursor = conn.cursor()

    hashed_password = hash_password(password)

    try:
        cursor.execute(
            "INSERT INTO users (username,email,password) VALUES (?,?,?)",
            (username, email, hashed_password)
        )

        conn.commit()
        return True

    except Exception:
        return False

    finally:
        conn.close()


# ===============================
# Login User
# ===============================

def login_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT user_id,password FROM users WHERE username=?",
        (username,)
    )

    result = cursor.fetchone()

    conn.close()

    if result:

        user_id, hashed = result

        if verify_password(password, hashed):
            return user_id

    return None