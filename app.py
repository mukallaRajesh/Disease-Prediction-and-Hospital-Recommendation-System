import os
import json
import logging
import sqlite3
from pathlib import Path
from collections import Counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import joblib

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    flash
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from google import genai


BASE_DIR = Path(__file__).resolve().parent

HOSPITALS_FILE = BASE_DIR / "HospitalsInIndia.csv"

CACHE_DIR = BASE_DIR / ".hf_cache"
CACHE_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)


app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "change-this-secret-key"
)

app.config["SESSION_TYPE"] = "filesystem"

logging.basicConfig(
    level=logging.DEBUG
)


# ============================================================
# GEMINI
# ============================================================

# Read the Gemini API key from the deployment environment.
# Set GEMINI_API_KEY in Render; never hard-code the key here.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

try:
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("Gemini client initialized successfully.")
    else:
        gemini_client = None
        logging.warning(
            "GEMINI_API_KEY is not configured. The chatbot will use dataset answers only."
        )
except Exception as e:
    logging.exception("Gemini client initialization error")
    gemini_client = None


# ============================================================
# DATABASE
# ============================================================

def init_db():

    db_path = BASE_DIR / "users.db"

    conn = sqlite3.connect(db_path)

    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users
        (
            email TEXT PRIMARY KEY,
            password TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            symptoms TEXT,
            predicted_disease TEXT,
            confidence FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email)
            REFERENCES users(email)
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_history
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            quiz_type TEXT,
            score INTEGER,
            result TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email)
            REFERENCES users(email)
        )
        """
    )

    conn.commit()
    conn.close()


init_db()


# ============================================================
# LOAD DATASETS AND MODELS
# ============================================================

try:

    logging.info(
        "Loading datasets and models..."
    )

    dis_sym_data_v1 = pd.read_csv(
        BASE_DIR / "Processed_Dataset1.csv"
    )

    doc_data = pd.read_csv(
        BASE_DIR / "Doctor_Versus_Disease.csv",
        encoding="latin1",
        names=[
            "Disease",
            "Specialist"
        ]
    )

    des_data = pd.read_csv(
        BASE_DIR / "Disease_Description.csv"
    )

    algorithms = joblib.load(
        BASE_DIR / "trained_algorithms.pkl"
    )

    le = joblib.load(
        BASE_DIR / "label_encoder.pkl"
    )

    chatbot_df = pd.read_csv(
        BASE_DIR / "chatbot.csv",
        encoding="cp1252"
    )

    chatbot_df["User Question"] = (
        chatbot_df["User Question"]
        .fillna("")
    )

    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    embeddings_file = BASE_DIR / "question_embeddings.pkl"
    if embeddings_file.exists():
        logging.info("Loading precomputed question embeddings...")
        loaded_emb = joblib.load(embeddings_file)
        if isinstance(loaded_emb, (list, tuple)):
            chatbot_df["question_embedding"] = list(loaded_emb)
        else:
            chatbot_df["question_embedding"] = [e for e in loaded_emb]
    else:
        logging.info("Encoding question embeddings...")
        embeddings = sentence_model.encode(
            chatbot_df["User Question"].tolist(),
            batch_size=64,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        chatbot_df["question_embedding"] = [emb for emb in embeddings]
        joblib.dump(chatbot_df["question_embedding"], embeddings_file)

    chatbot_df["Answer"] = (
        chatbot_df["AI Response"]
    )

    dis_sym_data_v1 = dis_sym_data_v1.loc[
        :,
        ~dis_sym_data_v1.columns.str.contains(
            "^Unnamed"
        )
    ]

    test_col = [
        col
        for col in dis_sym_data_v1.columns
        if col != "Disease"
    ]

    logging.info(
        "Datasets and models loaded successfully!"
    )

except Exception as e:

    logging.error(
        f"Error loading models or datasets: {e}"
    )

    raise


# ============================================================
# LOAD 1,348 HOSPITALS
# ============================================================

try:

    hospitals_df = pd.read_csv(
        HOSPITALS_FILE
    )

    hospitals_df["Hospital"] = (
        hospitals_df["Hospital"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    hospitals_df["State"] = (
        hospitals_df["State"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    hospitals_df["City"] = (
        hospitals_df["City"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    hospitals_df["LocalAddress"] = (
        hospitals_df["LocalAddress"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if "Pincode" in hospitals_df.columns:

        hospitals_df["Pincode"] = (
            hospitals_df["Pincode"]
            .fillna("")
            .astype(str)
            .str.replace(
                ".0",
                "",
                regex=False
            )
            .str.strip()
        )

    else:

        hospitals_df["Pincode"] = ""

    logging.info(
        f"Hospitals loaded: {len(hospitals_df)}"
    )

except Exception as e:

    logging.error(
        f"Could not load HospitalsInIndia.csv: {e}"
    )

    hospitals_df = pd.DataFrame()


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(f):

    def decorated_function(
        *args,
        **kwargs
    ):

        if "user" not in session:

            flash(
                "Please login to access this page.",
                "error"
            )

            return redirect(
                url_for("home")
            )

        return f(
            *args,
            **kwargs
        )

    decorated_function.__name__ = f.__name__

    return decorated_function


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if "user" in session:

        return redirect(
            url_for("predict_page")
        )

    return render_template(
        "landing.html"
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route("/signup")
def signup():

    if "user" in session:

        return redirect(
            url_for("predict_page")
        )

    return render_template(
        "signup.html"
    )


@app.route(
    "/signup",
    methods=["POST"]
)
def signup_post():

    email = request.form.get(
        "email"
    )

    password = request.form.get(
        "password"
    )

    confirm_password = request.form.get(
        "confirm_password"
    )

    if not email or not password:

        flash(
            "Please fill in all fields.",
            "error"
        )

        return redirect(
            url_for("signup")
        )

    if password != confirm_password:

        flash(
            "Passwords do not match.",
            "error"
        )

        return redirect(
            url_for("signup")
        )

    conn = sqlite3.connect(
        BASE_DIR / "users.db"
    )

    c = conn.cursor()

    c.execute(
        """
        SELECT email
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    if c.fetchone():

        flash(
            "Email already registered.",
            "error"
        )

        conn.close()

        return redirect(
            url_for("signup")
        )

    hashed_password = (
        generate_password_hash(
            password
        )
    )

    c.execute(
        """
        INSERT INTO users
        (email, password)
        VALUES (?, ?)
        """,
        (
            email,
            hashed_password
        )
    )

    conn.commit()
    conn.close()

    flash(
        "Registration successful! Please login.",
        "success"
    )

    return redirect(
        url_for("login_page")
    )


# ============================================================
# LOGIN
# ============================================================

@app.route("/login")
def login_page():

    if "user" in session:

        return redirect(
            url_for("predict_page")
        )

    return render_template(
        "login.html"
    )


@app.route(
    "/login",
    methods=["POST"]
)
def login():

    email = request.form.get(
        "email"
    )

    password = request.form.get(
        "password"
    )

    if not email or not password:

        flash(
            "Please fill in all fields.",
            "error"
        )

        return redirect(
            url_for("login_page")
        )

    conn = sqlite3.connect(
        BASE_DIR / "users.db"
    )

    c = conn.cursor()

    c.execute(
        """
        SELECT password
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    result = c.fetchone()

    conn.close()

    if result and check_password_hash(
        result[0],
        password
    ):

        session["user"] = email

        flash(
            "Login successful!",
            "success"
        )

        return redirect(
            url_for("predict_page")
        )

    flash(
        "Invalid email or password.",
        "error"
    )

    return redirect(
        url_for("login_page")
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.pop(
        "user",
        None
    )

    session.pop(
        "chat_history",
        None
    )

    session.pop(
        "location",
        None
    )

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# GET SYMPTOMS
# ============================================================

@app.route(
    "/get_symptoms",
    methods=["GET"]
)
@login_required
def get_symptoms():

    return jsonify({
        "symptoms": test_col
    })


# ============================================================
# LOCATION
# ============================================================

@app.route(
    "/location",
    methods=["GET", "POST"]
)
@app.route(
    "/location-page",
    methods=["GET"]
)
@login_required
def location_page():

    if request.method == "POST":
        try:

            data = request.get_json(
                silent=True
            ) or {}

            latitude = data.get(
                "latitude"
            )

            longitude = data.get(
                "longitude"
            )

            if latitude is None or longitude is None:

                return jsonify({
                    "error":
                    "Latitude and longitude are required."
                }), 400

            latitude = float(
                latitude
            )

            longitude = float(
                longitude
            )

            params = urlencode({

                "lat":
                latitude,

                "lon":
                longitude,

                "format":
                "jsonv2",

                "zoom":
                18,

                "addressdetails":
                1
            })

            url = (
                "https://nominatim.openstreetmap.org/reverse?"
                + params
            )

            req = Request(
                url,
                headers={
                    "User-Agent":
                    "Healthcare-Chatbot-Project/1.0"
                }
            )

            with urlopen(
                req,
                timeout=10
            ) as response:

                location_data = json.loads(
                    response
                    .read()
                    .decode("utf-8")
                )

            address = location_data.get(
                "address",
                {}
            )

            area = (
                address.get("suburb")
                or address.get("neighbourhood")
                or address.get("quarter")
                or address.get("residential")
                or address.get("village")
                or ""
            )

            city = (
                address.get("city")
                or address.get("town")
                or address.get("municipality")
                or ""
            )

            if not city:

                city = (
                    address.get("city_district")
                    or address.get("district")
                    or address.get("county")
                    or ""
                )

            state = (
                address.get("state")
                or ""
            )

            location_info = {

                "latitude":
                latitude,

                "longitude":
                longitude,

                "area":
                area,

                "city":
                city,

                "state":
                state
            }

            session["location"] = (
                location_info
            )

            session.modified = True

            logging.info(
                "Location detected: "
                f"{area}, {city}, {state}"
            )

            return jsonify({

                "success":
                True,

                "latitude":
                latitude,

                "longitude":
                longitude,

                "area":
                area,

                "city":
                city,

                "state":
                state,

                "display_name":
                location_data.get(
                    "display_name",
                    ""
                )
            })

        except Exception as e:

            logging.error(
                f"Location error: {e}"
            )

            return jsonify({

                "error":
                "Could not detect your area and city. "
                "Please allow location permission and try again."
            }), 500

    cities = []
    if not hospitals_df.empty:
        cities = sorted(
            set(
                str(city).strip()
                for city in hospitals_df["City"].dropna()
                if str(city).strip()
            )
        )

    return render_template(
        "location.html",
        cities=cities
    )


get_location = location_page


# ============================================================
# HOSPITAL SEARCH
# ============================================================

def get_recommended_hospitals(
    city,
    specialist="",
    area="",
    limit=10
):

    if hospitals_df.empty:
        return []

    if not city:
        return []

    city = str(
        city
    ).strip()

    area = str(
        area
    ).strip()

    city_lower = city.lower()

    city_df = hospitals_df[
        hospitals_df["City"]
        .str.lower()
        .str.strip()
        == city_lower
    ].copy()

    if city_df.empty:

        city_df = hospitals_df[
            hospitals_df["City"]
            .str.lower()
            .str.contains(
                city_lower,
                na=False,
                regex=False
            )
        ].copy()

    if city_df.empty:
        return []

    # First try the detected area.
    # If there are no hospitals in that area,
    # use the whole city.
    if area:

        area_lower = area.lower()

        area_df = city_df[
            city_df["LocalAddress"]
            .str.lower()
            .str.contains(
                area_lower,
                na=False,
                regex=False
            )
        ].copy()

        if not area_df.empty:

            city_df = area_df

    city_df = city_df.head(
        limit
    )

    hospitals = []

    for _, row in city_df.iterrows():

        hospital_name = row.get(
            "Hospital",
            ""
        )

        address = row.get(
            "LocalAddress",
            ""
        )

        hospital_city = row.get(
            "City",
            ""
        )

        hospital_state = row.get(
            "State",
            ""
        )

        pincode = row.get(
            "Pincode",
            ""
        )

        hospitals.append({

            "Hospital":
            str(hospital_name),

            "Address":
            str(address),

            "City":
            str(hospital_city),

            "State":
            str(hospital_state),

            "Pincode":
            str(pincode),

            "Specialist":
            str(specialist)
        })

    return hospitals


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/predict")
@login_required
def predict_page():

    cities = []

    if not hospitals_df.empty:

        cities = sorted(
            set(
                str(city).strip()
                for city
                in hospitals_df["City"].dropna()
                if str(city).strip()
            )
        )

    return render_template(
        "prediction.html",
        symptoms=test_col,
        cities=cities
    )


# ============================================================
# DIET RECOMMENDATIONS
# ============================================================

diet_recommendations = {

    "Malaria":
    "Focus on high-calorie foods to combat fatigue. Include fruits like bananas and oranges, leafy greens, and protein sources like chicken and fish.",

    "Allergy":
    "Avoid allergens based on individual sensitivities. Incorporate anti-inflammatory foods such as fatty fish, berries, and leafy greens.",

    "Hypothyroidism":
    "Increase iodine intake with iodized salt and seaweed. Include selenium-rich foods like Brazil nuts and zinc-rich foods such as pumpkin seeds.",

    "Psoriasis":
    "Follow an anti-inflammatory diet rich in omega-3 fatty acids, fruits, vegetables, and whole grains while limiting processed foods.",

    "GERD":
    "Eat smaller meals and avoid trigger foods like spicy dishes, chocolate, caffeine, and acidic fruits.",

    "Chronic cholestasis":
    "Limit saturated fats and include fiber-rich foods such as fruits, vegetables, and whole grains.",

    "hepatitis A":
    "Stay hydrated and consume a balanced diet rich in fruits, vegetables, lean proteins, and whole grains while avoiding alcohol.",

    "Osteoarthristis":
    "Focus on anti-inflammatory foods such as fatty fish, nuts, olive oil, and plenty of fruits and vegetables.",

    "(vertigo) Paroymsal  Positional Vertigo":
    "Stay hydrated and limit caffeine and alcohol. Include vitamin D-rich foods.",

    "Hypoglycemia":
    "Eat small frequent meals with complex carbohydrates, protein, and healthy fats.",

    "Acne":
    "Reduce dairy intake if it triggers symptoms and include antioxidant-rich foods and omega-3 fatty acids.",

    "Diabetes":
    "Focus on low glycemic index foods such as whole grains, legumes, vegetables, and lean proteins.",

    "Impetigo":
    "Include nutrient-dense foods and protein-rich foods along with fruits and vegetables.",

    "Hypertension":
    "Adopt a DASH-style diet rich in fruits, vegetables, whole grains, and lean protein while reducing sodium.",

    "Peptic ulcer diseae":
    "Avoid spicy foods and consume fiber-rich foods while eating smaller meals.",

    "Dimorphic hemorrhoids(piles)":
    "Increase fiber intake through fruits, vegetables, and whole grains and stay hydrated.",

    "Common Cold":
    "Stay hydrated and consume vitamin C-rich foods and zinc-rich foods.",

    "Chicken pox":
    "Maintain hydration and eat soft foods if mouth lesions are present.",

    "Cervical spondylosis":
    "Focus on a balanced diet with anti-inflammatory foods and adequate calcium.",

    "Hyperthyroidism":
    "Focus on a balanced diet with adequate protein from lean sources.",

    "Urinary tract infection":
    "Drink plenty of water and avoid bladder irritants such as caffeine and alcohol.",

    "Varicose veins":
    "Increase fiber intake and consume fruits and vegetables.",

    "AIDS":
    "Focus on nutrient-dense foods including lean proteins, whole grains, fruits, and vegetables.",

    "Paralysis (brain hemorrhage)":
    "Consume a balanced diet rich in fruits, vegetables, and omega-3 fatty acids.",

    "Typhoid":
    "Focus on easily digestible foods such as rice and bananas and stay hydrated.",

    "Hepatitis B":
    "Avoid alcohol and consume a balanced diet rich in fruits and vegetables.",

    "Fungal infection":
    "Limit excessive sugar intake and consume a balanced diet.",

    "Hepatitis C":
    "Follow a balanced diet rich in fruits and vegetables and avoid alcohol.",

    "Migraine":
    "Identify food triggers and maintain good hydration.",

    "Bronchial Asthma":
    "Avoid foods that trigger allergies and maintain a balanced diet.",

    "Alcoholic Hepatitis":
    "Avoid alcohol completely and consume a balanced nutrient-rich diet.",

    "Jaundice":
    "Avoid fatty or fried foods and increase fluid intake.",

    "Hepatitis E":
    "Stay hydrated and consume a balanced diet while avoiding alcohol.",

    "Dengue":
    "Stay hydrated and consume nutrient-dense foods.",

    "Pneumonia":
    "Stay hydrated and consume nutrient-dense foods that support recovery.",

    "Hepatitis D":
    "Avoid alcohol and eat a balanced diet rich in vitamins.",

    "Heart Attack":
    "Adopt a heart-healthy diet low in saturated fats and rich in whole grains, lean proteins, fruits, and vegetables.",

    "Arthritis":
    "Follow an anti-inflammatory diet rich in fruits, vegetables, fish, and nuts.",

    "Gastroenteritis":
    "Stay hydrated with clear fluids and consume bland foods until symptoms improve.",

    "Tuberculosis":
    "Focus on high-calorie nutrient-dense foods and adequate protein."
}


# ============================================================
# DOCTOR & SPECIALIST RECOMMENDATION PROFILES
# ============================================================

SPECIALIST_PROFILES = {
    "allergist": {
        "title": "Allergist / Immunologist",
        "description": "Specializes in diagnosing, managing, and treating allergic reactions, asthma, and immune system disorders.",
        "when_to_see": "Consult an Allergist if you suffer from severe allergic reactions, chronic nasal congestion, food sensitivities, or breathing difficulties triggered by allergens.",
        "icon": "fa-shield-virus"
    },
    "cardiologist": {
        "title": "Cardiologist (Heart Specialist)",
        "description": "Expert physician focused on diagnosing and treating diseases of the heart and blood vessels, including hypertension, chest pain, and heart disease.",
        "when_to_see": "Consult a Cardiologist for persistent high blood pressure, chest tightness, palpitations, irregular heartbeat, or breathlessness during activity.",
        "icon": "fa-heart-pulse"
    },
    "dermatologist": {
        "title": "Dermatologist (Skin Specialist)",
        "description": "Medical specialist focusing on conditions of the skin, hair, and nails, including fungal infections, acne, eczema, psoriasis, and dermatitis.",
        "when_to_see": "Consult a Dermatologist for persistent skin eruptions, severe acne, peeling skin, unexplained rashes, itching, or abnormal skin lesions.",
        "icon": "fa-hand-dots"
    },
    "endocrinologist": {
        "title": "Endocrinologist (Hormone & Diabetes Specialist)",
        "description": "Specialist in hormone-producing glands and metabolism, managing conditions like diabetes, thyroid imbalances, and adrenal disorders.",
        "when_to_see": "Consult an Endocrinologist for diabetes management, fluctuating blood sugar levels, thyroid swelling, sudden weight changes, or chronic fatigue.",
        "icon": "fa-dna"
    },
    "gastroenterologist": {
        "title": "Gastroenterologist (Digestive Specialist)",
        "description": "Specialist in gastrointestinal health, treating disorders of the stomach, intestines, esophagus, liver, pancreas, and bile ducts.",
        "when_to_see": "Consult a Gastroenterologist for chronic acid reflux (GERD), peptic ulcer pain, prolonged abdominal cramping, severe nausea, or digestive trouble.",
        "icon": "fa-utensils"
    },
    "gynecologist": {
        "title": "Gynecologist (Women's Health Specialist)",
        "description": "Physician specializing in female reproductive health, hormonal cycles, and gynecological wellness.",
        "when_to_see": "Consult a Gynecologist for pelvic discomfort, hormonal imbalances, irregular cycles, or routine female health examinations.",
        "icon": "fa-venus"
    },
    "hepatologist": {
        "title": "Hepatologist (Liver Specialist)",
        "description": "Super-specialist in liver diseases, viral hepatitis (A, B, C, E), alcoholic liver damage, jaundice, and gallbladder disorders.",
        "when_to_see": "Consult a Hepatologist for yellowing of eyes/skin (jaundice), dark urine, elevated liver enzymes, or suspected hepatitis infections.",
        "icon": "fa-flask-vial"
    },
    "internal medcine": {
        "title": "Internal Medicine Specialist",
        "description": "Physician specialized in broad-spectrum adult healthcare, chronic multisystem disease management, and in-depth clinical diagnosis.",
        "when_to_see": "Consult an Internal Medicine doctor when experiencing multiple simultaneous symptoms, unexplained fever, or requiring comprehensive clinical assessment.",
        "icon": "fa-stethoscope"
    },
    "neurologist": {
        "title": "Neurologist (Brain & Nerve Specialist)",
        "description": "Specialist treating disorders of the central and peripheral nervous system, including migraines, nerve pain, stroke, paralysis, and vertigo.",
        "when_to_see": "Consult a Neurologist for chronic or severe migraines, facial numbness, limb weakness, severe dizziness, balance loss, or seizures.",
        "icon": "fa-brain"
    },
    "osteopathic": {
        "title": "Orthopedic / Musculoskeletal Specialist",
        "description": "Expert in the musculoskeletal system, diagnosing conditions affecting bones, joints, spine, ligaments, and cervical vertebrae.",
        "when_to_see": "Consult an Orthopedic or Osteopathic specialist for chronic cervical neck pain, back stiffness, restricted mobility, or joint discomfort.",
        "icon": "fa-bone"
    },
    "otolaryngologist": {
        "title": "Otolaryngologist (ENT Specialist)",
        "description": "Specialist in medical and surgical management of disorders of the ear, nose, throat, sinuses, larynx, and related neck structures.",
        "when_to_see": "Consult an ENT specialist for persistent sinus pressure, ear infections, hearing loss, throat discomfort, or chronic nasal congestion.",
        "icon": "fa-ear-listen"
    },
    "pediatrician": {
        "title": "Pediatrician (Child Specialist)",
        "description": "Physician specializing in physical, behavioral, and developmental health of babies, children, and teenagers.",
        "when_to_see": "Consult a Pediatrician for infant and childhood fevers, pediatric rashes, vaccination counseling, or growth and developmental milestones.",
        "icon": "fa-baby"
    },
    "phlebologist": {
        "title": "Phlebologist (Vein Specialist)",
        "description": "Specialist dedicated to diagnosing and treating venous disorders such as varicose veins, spider veins, and venous insufficiency.",
        "when_to_see": "Consult a Phlebologist for bulging varicose veins, swollen lower legs, leg heaviness, or burning sensations in calf veins.",
        "icon": "fa-wave-square"
    },
    "pulmonologist": {
        "title": "Pulmonologist (Chest & Lung Specialist)",
        "description": "Specialist in diseases of the respiratory tract, lungs, and bronchial tubes, managing asthma, pneumonia, chronic cough, and bronchitis.",
        "when_to_see": "Consult a Pulmonologist for chronic coughing, wheezing, shortness of breath, lung congestion, or lingering respiratory infections.",
        "icon": "fa-lungs"
    },
    "rheumatologists": {
        "title": "Rheumatologist (Joint & Arthritis Specialist)",
        "description": "Specialist focused on autoimmune and inflammatory musculoskeletal diseases including osteoarthritis, rheumatoid arthritis, and gout.",
        "when_to_see": "Consult a Rheumatologist for morning joint stiffness, chronic swelling or warmth in knees/fingers, and progressive joint ache.",
        "icon": "fa-person-walking"
    },
    "tuberculosis": {
        "title": "Pulmonologist / Infectious Disease Specialist",
        "description": "Medical specialist focusing on mycobacterial tuberculosis, infectious respiratory illnesses, and targeted antimicrobial therapies.",
        "when_to_see": "Consult an Infectious Disease or Chest specialist for persistent cough lasting >2 weeks, coughing blood, night sweats, or unexplained weight loss.",
        "icon": "fa-virus"
    },
    "general physician": {
        "title": "General Physician",
        "description": "Primary care medical doctor providing comprehensive consultations, preventive care, first-line diagnoses, and specialist referrals.",
        "when_to_see": "Consult a General Physician for general symptoms, initial checkups, viral fevers, prescription renewals, and triage.",
        "icon": "fa-user-doctor"
    }
}


def get_specialist_info(specialist_name):
    if not specialist_name:
        return SPECIALIST_PROFILES["general physician"]
    clean = str(specialist_name).replace("\xa0", " ").strip().lower()
    if clean in SPECIALIST_PROFILES:
        return SPECIALIST_PROFILES[clean]
    for key, val in SPECIALIST_PROFILES.items():
        if key in clean or clean in key:
            return val
    return {
        "title": str(specialist_name).strip(),
        "description": "Medical specialist qualified to diagnose and treat conditions in this clinical domain.",
        "when_to_see": f"Consult a {specialist_name} for dedicated clinical evaluation, diagnostic tests, and tailored treatment plans.",
        "icon": "fa-user-md"
    }


def remove_allergens(
    diet_str,
    allergens
):

    for allergen in allergens:

        allergen = str(
            allergen
        ).lower()

        if allergen in diet_str.lower():

            diet_str = diet_str.replace(
                allergen,
                "Not Recommended due to allergy"
            )

    return diet_str


# ============================================================
# DISEASE PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    try:
        data = request.get_json(silent=True) or {}
        logging.debug(f"Received prediction data: {data}")

        symptoms = data.get("symptoms", [])
        if not isinstance(symptoms, list) or not symptoms:
            return jsonify({
                "error": "Please select at least 3 symptoms."
            }), 400

        # Location is optional (from payload or session)
        location = session.get("location", {})
        city = str(data.get("city", "")).strip() or str(location.get("city", "")).strip()
        area = str(data.get("area", "")).strip() or str(location.get("area", "")).strip()
        state = str(data.get("state", "")).strip() or str(location.get("state", "")).strip()
        latitude = data.get("latitude") or location.get("latitude")
        longitude = data.get("longitude") or location.get("longitude")

        # Format symptoms to match dataset column names
        formatted_symptoms = []
        for symptom in symptoms:
            s_clean = str(symptom).strip()
            if s_clean in test_col:
                formatted_symptoms.append(s_clean)
            else:
                s_norm = s_clean.lower().replace(" ", "_")
                s_norm = "".join(c for c in s_norm if c.isalnum() or c == "_")
                if s_norm in test_col:
                    formatted_symptoms.append(s_norm)

        if not formatted_symptoms:
            return jsonify({
                "error": "No valid symptoms matched."
            }), 400

        # Create ML input DataFrame
        test_data = {
            col: 1 if col in formatted_symptoms else 0
            for col in test_col
        }
        test_df = pd.DataFrame([test_data])

        # Voting ensemble
        predicted = []
        for model_name, values in algorithms.items():
            model = values["model"]
            model_prediction = model.predict(test_df)
            decoded_prediction = le.inverse_transform(model_prediction)
            predicted.extend(decoded_prediction)

        if not predicted:
            return jsonify({
                "error": "Unable to generate prediction."
            }), 500

        disease_counts = Counter(predicted)
        total_models = len(algorithms)

        percentage_per_disease = {
            disease: (count / total_models) * 100
            for disease, count in disease_counts.items()
        }

        result_df = pd.DataFrame({
            "Disease": list(percentage_per_disease.keys()),
            "Chances": list(percentage_per_disease.values())
        })

        result_df = result_df.sort_values(
            by="Chances",
            ascending=False
        ).reset_index(drop=True)

        result_df = result_df.merge(doc_data, on="Disease", how="left")
        result_df = result_df.merge(des_data, on="Disease", how="left")

        # Store top prediction in prediction_history
        top_prediction = result_df.iloc[0]
        try:
            conn = sqlite3.connect(BASE_DIR / "users.db")
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO prediction_history
                (
                    user_email,
                    symptoms,
                    predicted_disease,
                    confidence
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    session["user"],
                    ",".join(symptoms),
                    top_prediction["Disease"],
                    float(top_prediction["Chances"])
                )
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            logging.error(f"Error saving prediction history: {db_err}")

        # Fetch nearby hospitals if city is available
        city_hospitals = []
        if city:
            city_hospitals = get_recommended_hospitals(
                city=city,
                area=area,
                limit=10
            )

        # Build results list matching prediction.html requirements
        results = []
        for _, row in result_df.iterrows():
            disease_name = str(row["Disease"])
            raw_specialist = str(row.get("Specialist", "General Physician")).replace("\xa0", " ").strip()
            spec_info = get_specialist_info(raw_specialist)

            diet = diet_recommendations.get(
                disease_name,
                "Maintain a balanced, nutritious diet and stay well hydrated."
            )

            # Disease-specific hospital recommendations if city provided
            disease_hospitals = []
            if city:
                disease_hospitals = get_recommended_hospitals(
                    city=city,
                    specialist=spec_info.get("title", ""),
                    area=area,
                    limit=5
                )
            if not disease_hospitals and city_hospitals:
                disease_hospitals = city_hospitals[:5]

            results.append({
                "Disease": disease_name,
                "Chances": round(float(row["Chances"]), 2),
                "Specialist": spec_info.get("title", raw_specialist),
                "Specialist_Role": spec_info.get("description", "Qualified medical practitioner."),
                "Specialist_Advice": spec_info.get("when_to_see", "Consult for medical guidance."),
                "Specialist_Icon": spec_info.get("icon", "fa-user-doctor"),
                "Description": str(row.get("Description", "No description available")),
                "Diet Recommendation": diet,
                "City": city,
                "Area": area,
                "State": state,
                "Latitude": latitude,
                "Longitude": longitude,
                "Hospitals": disease_hospitals
            })

        logging.debug(f"Final Prediction Results count: {len(results)}")
        return jsonify(results)

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# QUIZ DATA
# ============================================================

QUIZ_DATA = {}


def create_quiz(
    title,
    icon,
    description,
    questions
):

    return {
        "title": title,
        "icon": icon,
        "description": description,
        "questions": questions
    }


QUIZ_DATA["heart"] = create_quiz(
    "Heart Health",
    "❤️",
    "Learn important facts about heart health and healthy habits.",
    [
        {
            "question": "Which organ pumps blood throughout the body?",
            "options": [
                "Lungs",
                "Heart",
                "Kidneys",
                "Liver"
            ],
            "answer": 1
        },
        {
            "question": "Which habit is generally good for heart health?",
            "options": [
                "Regular physical activity",
                "Smoking",
                "Avoiding all movement",
                "Eating only processed foods"
            ],
            "answer": 0
        },
        {
            "question": "Which food choice is generally heart-friendly?",
            "options": [
                "Fruits and vegetables",
                "Only fried foods",
                "Sugary drinks only",
                "Processed snacks only"
            ],
            "answer": 0
        },
        {
            "question": "Smoking can increase the risk of:",
            "options": [
                "Heart disease",
                "Better circulation",
                "Stronger lungs",
                "Improved blood pressure"
            ],
            "answer": 0
        },
        {
            "question": "Regular physical activity can help support:",
            "options": [
                "Heart health",
                "Smoking addiction",
                "Dehydration",
                "Poor sleep"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["lung"] = create_quiz(
    "Lung Health",
    "🫁",
    "Test your knowledge about lungs, breathing, and respiratory health.",
    [
        {
            "question": "What is the main function of the lungs?",
            "options": [
                "Digest food",
                "Exchange oxygen and carbon dioxide",
                "Pump blood",
                "Produce insulin"
            ],
            "answer": 1
        },
        {
            "question": "Which habit can damage lung health?",
            "options": [
                "Smoking",
                "Walking",
                "Drinking water",
                "Getting enough sleep"
            ],
            "answer": 0
        },
        {
            "question": "Which gas does the body need from the air?",
            "options": [
                "Oxygen",
                "Carbon monoxide",
                "Helium",
                "Methane"
            ],
            "answer": 0
        },
        {
            "question": "Good indoor ventilation can help improve:",
            "options": [
                "Air quality",
                "Bone density",
                "Blood sugar",
                "Vision"
            ],
            "answer": 0
        },
        {
            "question": "Which activity can support general respiratory fitness?",
            "options": [
                "Regular physical activity",
                "Smoking",
                "Avoiding all movement",
                "Constant exposure to smoke"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["diabetes"] = create_quiz(
    "Diabetes",
    "🩸",
    "Learn basic facts about blood sugar and diabetes management.",
    [
        {
            "question": "Diabetes is mainly related to problems with:",
            "options": [
                "Blood glucose regulation",
                "Hearing",
                "Hair growth",
                "Bone length"
            ],
            "answer": 0
        },
        {
            "question": "Which organ produces insulin?",
            "options": [
                "Pancreas",
                "Lungs",
                "Heart",
                "Stomach"
            ],
            "answer": 0
        },
        {
            "question": "Which food group can contain carbohydrates?",
            "options": [
                "Grains",
                "Water",
                "Salt only",
                "Plain air"
            ],
            "answer": 0
        },
        {
            "question": "Regular physical activity can help with:",
            "options": [
                "Blood glucose management",
                "Making diabetes disappear instantly",
                "Stopping all medication automatically",
                "Removing the need for monitoring"
            ],
            "answer": 0
        },
        {
            "question": "People with diabetes should follow:",
            "options": [
                "Advice from their healthcare professional",
                "Random internet advice only",
                "No monitoring",
                "A diet with no water"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["blood_pressure"] = create_quiz(
    "Blood Pressure",
    "💓",
    "Test your knowledge about blood pressure and healthy habits.",
    [
        {
            "question": "Blood pressure measures the force of blood against:",
            "options": [
                "Artery walls",
                "Bones",
                "Skin",
                "Hair"
            ],
            "answer": 0
        },
        {
            "question": "Which lifestyle habit can support healthy blood pressure?",
            "options": [
                "Regular physical activity",
                "Smoking",
                "Excessive salt intake",
                "Long-term inactivity"
            ],
            "answer": 0
        },
        {
            "question": "High blood pressure is sometimes called:",
            "options": [
                "Hypertension",
                "Hypothermia",
                "Anemia",
                "Insomnia"
            ],
            "answer": 0
        },
        {
            "question": "Which mineral is often associated with limiting intake for blood pressure management?",
            "options": [
                "Sodium",
                "Oxygen",
                "Iron",
                "Calcium only"
            ],
            "answer": 0
        },
        {
            "question": "A blood pressure measurement is commonly written as:",
            "options": [
                "Two numbers",
                "One letter",
                "A temperature only",
                "A weight only"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["brain"] = create_quiz(
    "Brain Health",
    "🧠",
    "Learn about the brain, memory, sleep, and healthy habits.",
    [
        {
            "question": "Which organ controls many functions of the nervous system?",
            "options": [
                "Brain",
                "Liver",
                "Kidney",
                "Stomach"
            ],
            "answer": 0
        },
        {
            "question": "Adequate sleep can support:",
            "options": [
                "Memory and concentration",
                "Dehydration",
                "Poor attention",
                "Hearing loss"
            ],
            "answer": 0
        },
        {
            "question": "Which activity can help keep the brain engaged?",
            "options": [
                "Learning new skills",
                "Avoiding all mental activity",
                "Chronic sleep deprivation",
                "Smoking"
            ],
            "answer": 0
        },
        {
            "question": "The brain is part of the:",
            "options": [
                "Nervous system",
                "Digestive system",
                "Skeletal system",
                "Respiratory system"
            ],
            "answer": 0
        },
        {
            "question": "A balanced lifestyle can support:",
            "options": [
                "Overall brain health",
                "Only hair growth",
                "Only height",
                "None of these"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["bone"] = create_quiz(
    "Bone Health",
    "🦴",
    "Test your knowledge about bones, calcium, and physical activity.",
    [
        {
            "question": "What gives the body structural support?",
            "options": [
                "Bones",
                "Lungs",
                "Hair",
                "Eyes"
            ],
            "answer": 0
        },
        {
            "question": "Which mineral is important for bone health?",
            "options": [
                "Calcium",
                "Sodium only",
                "Chlorine only",
                "Carbon monoxide"
            ],
            "answer": 0
        },
        {
            "question": "Which activity can support bone strength?",
            "options": [
                "Weight-bearing exercise",
                "Long-term inactivity",
                "Smoking",
                "Avoiding movement"
            ],
            "answer": 0
        },
        {
            "question": "Vitamin D helps the body absorb:",
            "options": [
                "Calcium",
                "Smoke",
                "Carbon monoxide",
                "Alcohol"
            ],
            "answer": 0
        },
        {
            "question": "Bones also help protect important:",
            "options": [
                "Organs",
                "Passwords",
                "Computer files",
                "Clothing"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["skin"] = create_quiz(
    "Skin Health",
    "🧴",
    "Learn basic facts about skin care and protection.",
    [
        {
            "question": "What is the body's largest organ?",
            "options": [
                "Skin",
                "Heart",
                "Liver",
                "Brain"
            ],
            "answer": 0
        },
        {
            "question": "What can help protect skin from excessive ultraviolet radiation?",
            "options": [
                "Sunscreen",
                "Smoking",
                "Ignoring sun exposure",
                "Burning the skin"
            ],
            "answer": 0
        },
        {
            "question": "Why is hydration important for the body?",
            "options": [
                "It supports normal body functions",
                "It removes the need for sleep",
                "It prevents every disease",
                "It replaces all nutrients"
            ],
            "answer": 0
        },
        {
            "question": "Regular gentle cleansing can help:",
            "options": [
                "Keep skin clean",
                "Guarantee no skin disease",
                "Change natural blood type",
                "Increase height"
            ],
            "answer": 0
        },
        {
            "question": "A new or changing skin lesion should be:",
            "options": [
                "Discussed with a healthcare professional",
                "Always ignored",
                "Scratched repeatedly",
                "Burned"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["digestive"] = create_quiz(
    "Digestive Health",
    "🥗",
    "Test your knowledge about digestion, fiber, and healthy eating.",
    [
        {
            "question": "Where does most nutrient absorption occur?",
            "options": [
                "Small intestine",
                "Lungs",
                "Heart",
                "Ear"
            ],
            "answer": 0
        },
        {
            "question": "Which nutrient is commonly associated with healthy digestion?",
            "options": [
                "Fiber",
                "Smoke",
                "Alcohol only",
                "Carbon monoxide"
            ],
            "answer": 0
        },
        {
            "question": "Which foods are commonly good sources of fiber?",
            "options": [
                "Fruits, vegetables, and whole grains",
                "Only sugary drinks",
                "Only fried foods",
                "Only candy"
            ],
            "answer": 0
        },
        {
            "question": "Which organ helps produce bile?",
            "options": [
                "Liver",
                "Heart",
                "Lungs",
                "Brain"
            ],
            "answer": 0
        },
        {
            "question": "Adequate water intake can support:",
            "options": [
                "Normal digestion",
                "Bone shortening",
                "Hearing",
                "Vision only"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["eye"] = create_quiz(
    "Eye Health",
    "👁️",
    "Learn about vision, eye protection, and healthy habits.",
    [
        {
            "question": "Which organ is responsible for vision?",
            "options": [
                "Eye",
                "Ear",
                "Heart",
                "Kidney"
            ],
            "answer": 0
        },
        {
            "question": "Which nutrient is associated with normal vision?",
            "options": [
                "Vitamin A",
                "Carbon monoxide",
                "Alcohol",
                "Caffeine only"
            ],
            "answer": 0
        },
        {
            "question": "What can help protect eyes from excessive bright sunlight?",
            "options": [
                "UV-protective sunglasses",
                "Looking directly at the sun",
                "Ignoring sunlight",
                "Smoking"
            ],
            "answer": 0
        },
        {
            "question": "Regular eye examinations can help:",
            "options": [
                "Detect some eye problems early",
                "Guarantee perfect vision forever",
                "Change eye color",
                "Increase height"
            ],
            "answer": 0
        },
        {
            "question": "When using screens, regular breaks can help reduce:",
            "options": [
                "Eye strain",
                "Bone fractures",
                "Blood type changes",
                "Hair loss"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["dental"] = create_quiz(
    "Dental Health",
    "🦷",
    "Test your knowledge about teeth, gums, and oral hygiene.",
    [
        {
            "question": "How often is brushing generally recommended?",
            "options": [
                "Twice a day",
                "Once a month",
                "Once a year",
                "Never"
            ],
            "answer": 0
        },
        {
            "question": "Fluoride toothpaste helps protect against:",
            "options": [
                "Tooth decay",
                "Broken bones",
                "Hearing loss",
                "Eye strain"
            ],
            "answer": 0
        },
        {
            "question": "Which habit supports gum health?",
            "options": [
                "Regular brushing and cleaning between teeth",
                "Smoking",
                "Avoiding dental care",
                "Eating only candy"
            ],
            "answer": 0
        },
        {
            "question": "Regular dental visits can help detect:",
            "options": [
                "Dental problems",
                "All diseases",
                "All infections in the body",
                "Blood type"
            ],
            "answer": 0
        },
        {
            "question": "Frequent sugary foods and drinks can increase the risk of:",
            "options": [
                "Tooth decay",
                "Better enamel",
                "Stronger teeth",
                "Improved vision"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["nutrition"] = create_quiz(
    "Nutrition",
    "🥗",
    "Learn the basics of balanced nutrition and healthy eating.",
    [
        {
            "question": "A balanced diet should include:",
            "options": [
                "A variety of nutrient-rich foods",
                "Only sweets",
                "Only fried foods",
                "Only soft drinks"
            ],
            "answer": 0
        },
        {
            "question": "Which food is generally a good source of protein?",
            "options": [
                "Beans",
                "Soft drinks",
                "Candy",
                "Table sugar"
            ],
            "answer": 0
        },
        {
            "question": "Which nutrient provides a major source of energy?",
            "options": [
                "Carbohydrates",
                "Water only",
                "Minerals only",
                "Vitamins only"
            ],
            "answer": 0
        },
        {
            "question": "Which drink is essential for hydration?",
            "options": [
                "Water",
                "Only energy drinks",
                "Only soda",
                "Only sugary juice"
            ],
            "answer": 0
        },
        {
            "question": "Eating a variety of fruits and vegetables provides:",
            "options": [
                "Many vitamins, minerals, and other nutrients",
                "Only sugar",
                "Only fat",
                "No nutrients"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["immunity"] = create_quiz(
    "Immunity",
    "🛡️",
    "Test your knowledge about the immune system and healthy habits.",
    [
        {
            "question": "What is a major role of the immune system?",
            "options": [
                "Help defend the body against harmful pathogens",
                "Pump blood",
                "Digest all food",
                "Control height"
            ],
            "answer": 0
        },
        {
            "question": "Which habit supports general immune health?",
            "options": [
                "Adequate sleep",
                "Chronic sleep deprivation",
                "Smoking",
                "Poor nutrition"
            ],
            "answer": 0
        },
        {
            "question": "Vaccines help the immune system:",
            "options": [
                "Prepare to recognize specific infections",
                "Become unnecessary",
                "Stop all diseases",
                "Replace sleep"
            ],
            "answer": 0
        },
        {
            "question": "Good hand hygiene can help reduce:",
            "options": [
                "Spread of some infections",
                "Bone fractures",
                "Eye color changes",
                "Height"
            ],
            "answer": 0
        },
        {
            "question": "A healthy immune system is supported by:",
            "options": [
                "Balanced nutrition and healthy habits",
                "Smoking",
                "No sleep",
                "Extreme diets"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["wellness"] = create_quiz(
    "General Wellness",
    "🌿",
    "Test your knowledge about healthy daily habits and wellness.",
    [
        {
            "question": "Which habit supports overall wellness?",
            "options": [
                "Regular physical activity",
                "Smoking",
                "Chronic sleep deprivation",
                "Skipping all meals"
            ],
            "answer": 0
        },
        {
            "question": "Adults generally benefit from adequate:",
            "options": [
                "Sleep",
                "Smoke exposure",
                "Sugary drinks",
                "Screen time without breaks"
            ],
            "answer": 0
        },
        {
            "question": "Why is hydration important?",
            "options": [
                "Water supports many body functions",
                "It replaces all food",
                "It prevents every illness",
                "It replaces sleep"
            ],
            "answer": 0
        },
        {
            "question": "Stress management can be supported by:",
            "options": [
                "Relaxation, exercise, and social support",
                "Avoiding sleep",
                "Smoking",
                "Never taking breaks"
            ],
            "answer": 0
        },
        {
            "question": "Preventive healthcare can include:",
            "options": [
                "Recommended checkups and screenings",
                "Ignoring symptoms",
                "Avoiding all healthcare",
                "Self-diagnosing every condition"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["fever"] = create_quiz(
    "Fever",
    "🌡️",
    "Learn basic facts about fever and when medical care may be needed.",
    [
        {
            "question": "Fever is generally a temporary increase in:",
            "options": [
                "Body temperature",
                "Height",
                "Bone length",
                "Hair growth"
            ],
            "answer": 0
        },
        {
            "question": "Fever can occur as part of the body's response to:",
            "options": [
                "Some infections",
                "Only broken bones",
                "Only eye problems",
                "Only dental problems"
            ],
            "answer": 0
        },
        {
            "question": "During fever, maintaining adequate hydration is generally:",
            "options": [
                "Important",
                "Always harmful",
                "Unnecessary",
                "Impossible"
            ],
            "answer": 0
        },
        {
            "question": "A person with severe or concerning symptoms during a fever should:",
            "options": [
                "Seek medical advice",
                "Always ignore symptoms",
                "Exercise intensely",
                "Avoid all fluids"
            ],
            "answer": 0
        },
        {
            "question": "Temperature can be measured using a:",
            "options": [
                "Thermometer",
                "Stethoscope only",
                "Ruler",
                "Weighing scale"
            ],
            "answer": 0
        }
    ]
)


QUIZ_DATA["ear"] = create_quiz(
    "Ear Health",
    "👂",
    "Test your knowledge about hearing and healthy ear care.",
    [
        {
            "question": "Which organ is responsible for hearing?",
            "options": [
                "Ear",
                "Eye",
                "Heart",
                "Liver"
            ],
            "answer": 0
        },
        {
            "question": "Very loud sounds can damage:",
            "options": [
                "Hearing",
                "Bone length",
                "Blood type",
                "Hair color"
            ],
            "answer": 0
        },
        {
            "question": "A safe approach to ear cleaning is:",
            "options": [
                "Avoid inserting objects deep into the ear",
                "Use sharp objects",
                "Insert cotton swabs deeply",
                "Scratch the ear canal"
            ],
            "answer": 0
        },
        {
            "question": "Persistent hearing difficulty should be:",
            "options": [
                "Evaluated by a healthcare professional",
                "Ignored forever",
                "Treated with random medicines",
                "Exposed to louder sounds"
            ],
            "answer": 0
        },
        {
            "question": "Protective hearing equipment can be useful in:",
            "options": [
                "Very noisy environments",
                "Quiet rooms only",
                "Sleeping only",
                "Reading books"
            ],
            "answer": 0
        }
    ]
)


# ============================================================
# QUIZ ROUTES
# ============================================================

@app.route("/heart_quiz")
@login_required
def heart_quiz():
    return redirect(url_for("quiz_page", quiz_id="heart"))

@app.route("/lung_quiz")
@login_required
def lung_quiz():
    return redirect(url_for("quiz_page", quiz_id="lung"))

@app.route("/quiz")
@login_required
def quiz_center():

    return render_template(
        "quiz.html",
        quizzes=QUIZ_DATA
    )


@app.route("/quiz/<quiz_id>")
@login_required
def quiz_page(
    quiz_id
):

    if quiz_id not in QUIZ_DATA:

        flash(
            "Quiz not found.",
            "error"
        )

        return redirect(
            url_for("quiz_center")
        )

    quiz = QUIZ_DATA[
        quiz_id
    ]

    return render_template(
        "quiz_detail.html",
        quiz_id=quiz_id,
        quiz=quiz
    )


@app.route(
    "/quiz/<quiz_id>/submit",
    methods=["POST"]
)
@login_required
def quiz_submit(
    quiz_id
):

    if quiz_id not in QUIZ_DATA:

        flash(
            "Quiz not found.",
            "error"
        )

        return redirect(
            url_for("quiz_center")
        )

    quiz = QUIZ_DATA[
        quiz_id
    ]

    questions = quiz[
        "questions"
    ]

    score = 0

    for index, question in enumerate(
        questions
    ):

        selected_answer = request.form.get(
            f"question_{index}"
        )

        if selected_answer is not None:

            try:

                selected_answer = int(
                    selected_answer
                )

                if selected_answer == question[
                    "answer"
                ]:

                    score += 1

            except ValueError:

                pass

    total = len(
        questions
    )

    percentage = round(
        (score / total) * 100
    )

    if percentage >= 80:

        result = "Excellent"

    elif percentage >= 60:

        result = "Good"

    elif percentage >= 40:

        result = "Keep Learning"

    else:

        result = "Needs Improvement"

    quiz_name = (
        quiz["title"]
        + " Quiz"
    )

    conn = sqlite3.connect(
        BASE_DIR / "users.db"
    )

    c = conn.cursor()

    c.execute(
        """
        INSERT INTO quiz_history
        (
            user_email,
            quiz_type,
            score,
            result
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            session["user"],
            quiz_name,
            score,
            result
        )
    )

    conn.commit()
    conn.close()

    return render_template(
        "quiz_result.html",
        quiz_id=quiz_id,
        quiz=quiz,
        score=score,
        total=total,
        percentage=percentage,
        result=result
    )


# ============================================================
# CHATBOT
# ============================================================

@app.route("/chatbot")
@login_required
def chatbot():

    if "chat_history" not in session:

        session["chat_history"] = []

    cities = []
    if not hospitals_df.empty:
        cities = sorted(
            set(
                str(city).strip()
                for city in hospitals_df["City"].dropna()
                if str(city).strip()
            )
        )

    return render_template(
        "chatbot.html",
        chat_history=session[
            "chat_history"
        ],
        cities=cities
    )


def find_most_similar_answer(
    user_question
):

    user_question_embedding = (
        sentence_model.encode(
            user_question,
            convert_to_tensor=True
        )
    )

    similarities = [

        util.pytorch_cos_sim(
            user_question_embedding,
            emb
        ).item()

        for emb
        in chatbot_df[
            "question_embedding"
        ]
    ]

    most_similar_idx = similarities.index(
        max(similarities)
    )

    most_similar_row = chatbot_df.iloc[
        most_similar_idx
    ]

    return (
        most_similar_row[
            "User Question"
        ],

        most_similar_row[
            "Answer"
        ]
    )


def generate_gemini_response(
    chat_history,
    relevant_question,
    relevant_answer,
    city="",
    area=""
):

    prompt = (
        "You are a professional healthcare chatbot. "
        "Answer the user's healthcare question clearly, "
        "concisely, and professionally.\n\n"
    )

    if city:
        if area:
            prompt += f"User's specified location: {area}, {city}\n\n"
        else:
            prompt += f"User's specified location: {city}\n\n"

    prompt += (
        "Conversation History:\n"
    )

    for entry in chat_history:

        prompt += (
            f"User: {entry['user']}\n"
            f"Bot: {entry['bot']}\n"
        )

    prompt += (

        f"\nUser's latest question: "
        f"{chat_history[-1]['user']}\n\n"

        f"Relevant healthcare question: "
        f"{relevant_question}\n\n"

        f"Relevant healthcare information: "
        f"{relevant_answer}\n\n"

        "Instructions:\n"

        "- Answer directly based on the relevant healthcare information.\n"

        "- Keep the response concise and useful.\n"

        "- Use simple language.\n"

        "- Respond in 2-3 clear points when appropriate.\n"

        "- Do not unnecessarily repeat the question.\n"

        "- Do not invent medical information.\n"

        "- If the user asks about local healthcare, clinics, or hospitals, refer to their location.\n"

        "- If the information is insufficient, clearly say so.\n"

        "- For serious or emergency symptoms, advise the user "
        "to seek professional medical care immediately.\n"

        "- Mention a reference URL if one is available "
        "in the provided information.\n\n"

        "Bot:"
    )

    if gemini_client is None:

        return relevant_answer

    try:

        response = (
            gemini_client
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        )

        if response.text:

            return response.text.strip()

        return relevant_answer

    except Exception as e:

        logging.error(
            f"Gemini API error: {e}"
        )

        return relevant_answer


@app.route(
    "/chatbot/ask",
    methods=["POST"]
)
@login_required
def chatbot_ask():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "error":
                "Invalid request data"
            }), 400

        question = data.get(
            "question"
        )

        if not question:

            return jsonify({
                "error":
                "No question provided"
            }), 400

        question = question.strip()

        if not question:

            return jsonify({
                "error":
                "No question provided"
            }), 400

        city = str(data.get("city", "")).strip()
        area = str(data.get("area", "")).strip()

        if "chat_history" not in session:

            session["chat_history"] = []

        relevant_question, relevant_answer = (
            find_most_similar_answer(
                question
            )
        )

        session[
            "chat_history"
        ].append({

            "user":
            question,

            "bot":
            relevant_answer
        })

        enhanced_answer = (
            generate_gemini_response(

                session[
                    "chat_history"
                ],

                relevant_question,

                relevant_answer,

                city=city,

                area=area
            )
        )

        session[
            "chat_history"
        ][-1]["bot"] = (
            enhanced_answer
        )

        session.modified = True

        return jsonify({

            "response":
            enhanced_answer
        })

    except Exception as e:

        logging.error(
            f"Error in chatbot: {e}"
        )

        return jsonify({
            "error":
            str(e)
        }), 500


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
@login_required
def profile():

    conn = sqlite3.connect(
        BASE_DIR / "users.db"
    )

    c = conn.cursor()

    c.execute(
        """
        SELECT
            predicted_disease,
            confidence,
            timestamp,
            symptoms
        FROM prediction_history
        WHERE user_email = ?
        ORDER BY timestamp DESC
        """,
        (
            session["user"],
        )
    )

    predictions = c.fetchall()

    c.execute(
        """
        SELECT
            predicted_disease,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM prediction_history
        WHERE user_email = ?
        GROUP BY predicted_disease
        """,
        (
            session["user"],
        )
    )

    stats = c.fetchall()

    c.execute(
        """
        SELECT
            quiz_type,
            score,
            result,
            timestamp
        FROM quiz_history
        WHERE user_email = ?
        ORDER BY timestamp DESC
        """,
        (
            session["user"],
        )
    )

    quiz_history = c.fetchall()

    c.execute(
        """
        SELECT
            quiz_type,
            COUNT(*) as count,
            AVG(score) as avg_score
        FROM quiz_history
        WHERE user_email = ?
        GROUP BY quiz_type
        """,
        (
            session["user"],
        )
    )

    quiz_stats = c.fetchall()

    conn.close()

    return render_template(
        "profile.html",
        predictions=predictions,
        stats=stats,
        quiz_history=quiz_history,
        quiz_stats=quiz_stats,
        user_email=session["user"]
    )


# ============================================================
# RUN
# ============================================================
@app.route("/nearby-hospitals")
@login_required
def nearby_hospitals():
    city = request.args.get("city", "").strip()
    area = request.args.get("area", "").strip()

    hospitals = get_recommended_hospitals(
        city=city,
        specialist="",
        area=area,
        limit=10
    )

    return jsonify({
        "success": True,
        "hospitals": hospitals
    })
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )