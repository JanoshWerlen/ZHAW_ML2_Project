import os
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, g
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')

DATABASE = 'chatbot.db'


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def setup_database():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    last_login DATETIME)''')

        c.execute('''CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        message TEXT,
                        response TEXT,
                        action TEXT,
                        articles TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        rag_tokens_used INTEGER,
                        FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        conn.commit()




def register_user(username, password):
    db = get_db()
    hashed_password = generate_password_hash(password)
    try:
        db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        db.commit()
        print("Registration successful!")
    except sqlite3.IntegrityError:
        print("Username already exists. Please choose a different username.")

def authenticate_user(username, password):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if user and check_password_hash(user['password'], password):
        print("Login successful!")
        user_id = user['id']
        # Update the last_login timestamp
        db.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_id))
        db.commit()
        return user_id
    else:
        print("Invalid username or password.")
        return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_id = authenticate_user(username, password)
        if user_id:
            session['user_id'] = user_id
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials', 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            register_user(username, password)
            user_id = authenticate_user(username, password)
            if user_id:
                session['user_id'] = user_id
                return redirect(url_for('index'))
        except Exception as e:
            return str(e), 400
    return render_template('register.html')


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        user_id = session['user_id']
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        
        # Fetch the user from the database
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

        if user and check_password_hash(user['password'], current_password):
            # Check if the new password is no more than 8 characters long
            if len(new_password) > 8:
                return "New password must be no more than 8 characters long.", 400

            # Update the password
            hashed_password = generate_password_hash(new_password)
            db.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
            db.commit()
            return redirect(url_for('index'))
        else:
            return "Current password is incorrect.", 400

    return render_template('change_password.html')


@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    
    return redirect(url_for('login'))
