import json
import os
import sqlite

# --- File Paths ---
USERS_FILE = 'users.json'
RESULTS_FILE = 'results_db.json'

# --- Load or initialize JSON data ---
def load_json(filename):
    if not os.path.exists(filename):
        return [] if filename == USERS_FILE else {}
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# --- Signup ---
def signup():
    name = input("Enter name: ")
    email = input("Enter email: ")
    password = input("Enter password: ")
    join_date = input("Enter join date (YYYY-MM-DD): ")

    users = load_json(USERS_FILE)

    for user in users:
        if user['email'] == email:
            print("User already exists!")
            return

    new_user = {
        "name": name,
        "email": email,
        "password": password,
        "join_date": join_date
    }
    users.append(new_user)
    save_json(users, USERS_FILE)
    print("Signup successful!")

# --- Login ---
def login():
    email = input("Enter email: ")
    password = input("Enter password: ")

    users = load_json(USERS_FILE)
    for user in users:
        if user['email'] == email and user['password'] == password:
            print("Login successful!")
            return email
    print("Invalid credentials!")
    return None

# --- Add Try-On Result ---
def save_result(user_email):
    image_name = input("Enter image filename to save (e.g., finalimg.png): ")
    results = load_json(RESULTS_FILE)

    if user_email not in results:
        results[user_email] = []
    results[user_email].append(image_name)

    save_json(results, RESULTS_FILE)
    print("Result saved for user:", user_email)

# --- Main Menu ---
def main():
    while True:
        print("\n1. Signup")
        print("2. Login")
        print("3. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            signup()
        elif choice == '2':
            user_email = login()
            if user_email:
                while True:
                    print("\n--- Logged In Menu ---")
                    print("1. Save Try-On Result")
                    print("2. Logout")
                    sub_choice = input("Choose an option: ")
                    if sub_choice == '1':
                        save_result(user_email)
                    elif sub_choice == '2':
                        break
        elif choice == '3':
            break
        else:
            print("Invalid choice!")

if __name__ == '__main__':
    main()
