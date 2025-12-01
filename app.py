from flask import Flask, render_template, request, jsonify, url_for, session, redirect
import os
import json
import uuid
from datetime import datetime
from PIL import Image
import traceback
from gradio_client import Client, handle_file


app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['OUTPUT_FOLDER'] = 'static/output'
USER_FILE = 'users.json'
RESULTS_DB = 'results_db.json'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ------------------ JSON Helpers ------------------ #
def load_users():
    if not os.path.exists(USER_FILE):
        return []
    with open(USER_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def find_user_by_email(email):
    users = load_users()
    return next((u for u in users if u['email'] == email), None)

def load_results():
    if not os.path.exists('tryon_results.json'):
        return {}
    with open('tryon_results.json', 'r') as f:
        return json.load(f)

def save_result_for_user(email, image_path):
    db = load_results()
    db.setdefault(email, []).append(image_path)
    with open('tryon_results.json', 'w') as f:
        json.dump(db, f, indent=2)


# ------------------ Routes ------------------ #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/products')
def products():
    return render_template('products.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/trynow')
def trynow():
    return render_template('trynow.html')

@app.route('/single-product')
def single_product():
    return render_template('single-product.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/results')
def results():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    db = load_results()
    images = db.get(user, [])
    return render_template('results.html', images=images)

@app.route('/skin_predictor')
def skin_predictor():
    return render_template('skin_predictor.html')

@app.route('/ai_assistant')
def assistant():
    return render_template('ai_assistant.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = find_user_by_email(email)
        if user and user['password'] == password:
            session['user'] = user['email']
            return redirect(url_for('profile'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if find_user_by_email(email):
            return "Email already registered"

        users = load_users()
        users.append({
            "name": name,
            "email": email,
            "password": password,
            "join_date": datetime.now().strftime("%Y-%m-%d")
        })
        save_users(users)
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    user_email = session['user']
    user = find_user_by_email(user_email)
    db = load_results()
    images = db.get(user_email, [])
    return render_template('profile.html', user=user, tryon_history=images)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
def upload():
    try:
        user_img = request.files.get('user_image')
        garment_img = request.files.get('garment_image')

        if not user_img:
            return jsonify({"error": "User image is required"}), 400
        if not garment_img:
            return jsonify({"error": "Garment image is required"}), 400

        # Create temp directory
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Save uploaded files
        user_img_path = os.path.join("temp", f"user_{timestamp}.jpg")
        garment_img_path = os.path.join("temp", f"garment_{timestamp}.jpg")
        user_img.save(user_img_path)
        garment_img.save(garment_img_path)

        # Call Hugging Face Gradio client
        client = Client("yisol/IDM-VTON")
        try:
            result = client.predict(
                dict={"background": handle_file(user_img_path), "layers": [], "composite": None},
                garm_img=handle_file(garment_img_path),
                garment_des="Trying new outfit",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
        except Exception as e:
            print("Gradio API call failed:", e)
            return jsonify({"error": "The virtual try-on service failed. Please try again later."}), 500

        # Result handling (usually tuple)
        if isinstance(result, tuple) and isinstance(result[0], str) and result[0].startswith("http"):
            return jsonify({"processed_image_url": result[0]})
        elif isinstance(result, str) and result.startswith("http"):
            return jsonify({"processed_image_url": result})
        else:
            return jsonify({"error": "Unexpected response format from try-on API."}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/save_tryon', methods=['POST'])
def save_tryon():
    if 'user' not in session:
        return redirect(url_for('login'))

    image = request.files.get('result_image')
    if not image:
        return "No image uploaded", 400

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"result_{timestamp}.jpg"
    filepath = os.path.join('static', 'results', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    image.save(filepath)

    # Save to user history
    user_email = session['user']
    image_url = url_for('static', filename=f'results/{filename}')
    save_result_for_user(user_email, image_url)

    return redirect(url_for('profile'))

@app.route('/delete_image', methods=['POST'])
def delete_image():
    if 'user' not in session:
        return redirect(url_for('login'))

    image_url = request.form.get('image_url')
    user_email = session['user']
    db = load_results()

    if user_email in db and image_url in db[user_email]:
        # Remove from user list
        db[user_email].remove(image_url)

        # Remove actual file
        try:
            filename = image_url.split('/static/')[-1]
            file_path = os.path.join('static', filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete file: {e}")

        # Save updated JSON
        with open('tryon_results.json', 'w') as f:
            json.dump(db, f, indent=2)

    return redirect(url_for('profile'))


# ------------------ Run ------------------ #
if __name__ == '__main__':
    app.run(debug=True)
