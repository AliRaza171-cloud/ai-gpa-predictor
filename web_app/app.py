from flask import Flask, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "secret_key_123"

# ---------------- DATABASE ----------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
db = SQLAlchemy(app)

# ---------------- USER MODEL ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("gpa_model.pkl")
features = joblib.load("features.pkl")

# ---------------- HOME ----------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        # check duplicate user
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists!", "error")
            return redirect("/register")

        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully!", "success")
        return redirect("/login")

    return render_template("register.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user"] = username
            return redirect("/")
        else:
            flash("Invalid username or password", "error")
            return redirect("/login")

    return render_template("login.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

# ---------------- PREDICT ----------------
RETRAIN_LIMIT = 50
NEW_DATA_FILE = "new_data.csv"

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect("/login")

    data = request.form.to_dict()

    # ---------------- VALIDATION ----------------
    for key, value in data.items():
        if value == "":
            flash("Please fill all required fields!", "error")
            return redirect("/")

    # ---------------- PREDICTION ----------------
    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)[0]

    # ---------------- SAVE DATA ----------------
    new_data = pd.DataFrame([data])
    new_data["GPA_Post_AI"] = prediction

    file_exists = os.path.exists(NEW_DATA_FILE)
    new_data.to_csv(
        NEW_DATA_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )

    # ---------------- AUTO RETRAIN ----------------
    if file_exists:
        total_rows = len(pd.read_csv(NEW_DATA_FILE))

        if total_rows % RETRAIN_LIMIT == 0:
            os.system("python retrain.py")

    flash(f"Predicted GPA: {round(prediction, 2)}", "success")
    return redirect("/")

# ---------------- INIT DB ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)