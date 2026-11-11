# AI Disease Prediction & Hospital Recommendation System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Google Gemini API](https://img.shields.io/badge/Gemini_API-3.5_Flash_Lite-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Render](https://img.shields.io/badge/Deploy-Render_Ready-46E3B7?logo=render&logoColor=white)](https://render.com/)

An AI-powered healthcare platform combining **multi-model disease prediction**, **specialist doctor recommendations**, **nearby Indian hospital discovery**, a **3D interactive AI chatbot**, and a **15-topic multi-health quiz center**.

---

## 🌟 Key Features

### 1. 🩺 AI Disease Prediction Ensemble
* **Multi-Model Voting Ensemble:** Predicts health conditions by aggregating predictions across 6 trained classification algorithms:
  * Random Forest Classifier
  * Support Vector Classifier (SVC)
  * Decision Tree Classifier
  * Logistic Regression
  * Gaussian Naive Bayes
  * K-Nearest Neighbors (KNN)
* **Confidence & Probability Scoring:** Generates confidence percentages for matching diseases.
* **Clinical Descriptions & Dietary Guidance:** Provides targeted disease overviews and condition-specific nutrition recommendations.

### 2. 👨‍⚕️ Doctor & Specialist Recommendation System
* **Automated Specialist Matching:** Automatically maps predicted diseases to 18 distinct medical specialties (Cardiologist, Dermatologist, Pulmonologist, Neurologist, Otolaryngologist/ENT, Endocrinologist, Gastroenterologist, Rheumatologist, etc.).
* **Clinical Expertise Guidance:** Outlines the specialist's role and domain focus.
* **"When to Consult" Advice:** Practical clinical guidance helping users determine when to seek immediate medical consultation.

### 3. 🏥 Indian Hospital Recommendation System
* **Comprehensive Hospital Directory:** Built-in dataset of **1,348 hospitals across 298 Indian cities and states** (`HospitalsInIndia.csv`).
* **Facility Details:** Displays hospital name, street address, city, state, and PIN code.
* **One-Click Navigation:** Direct **"Directions"** button that opens Google Maps with the facility location.
* **Flexible Location Matching:** Detects city or area, with non-blocking fallback so predictions work seamlessly even without location input.

### 4. 🤖 AI Healthcare Chatbot with 3D Avatar
* **Interactive 3D Avatar:** Powered by Google `<model-viewer>` with real-time speech and idle animations.
* **Hybrid Intelligence:**
  * **Semantic Search:** Sentence-Transformers (`paraphrase-MiniLM-L6-v2`) identifies relevant verified medical Q&A entries in milliseconds.
  * **Gemini AI Generation:** Google Gemini (`gemini-3.5-flash-lite`) refines answers with contextual, compassionate, and accurate healthcare explanations.
* **Voice & Speech Support:** Built-in Web Speech API recognition for voice queries.

### 5. 📋 15-Topic Multi-Health Quiz Center
Engaging, educational assessments with real-time scoring across 15 vital wellness categories:
* ❤️ Heart Health
* 🫁 Lung Health
* 🩸 Diabetes & Blood Sugar
* 🩺 Blood Pressure & Hypertension
* 🧠 Brain & Neurological Wellness
* 🦴 Bone & Joint Health
* 🧴 Skin & Dermatological Care
* 🥗 Nutrition & Diet
* 👁️ Eye Health
* 🦷 Dental & Oral Hygiene
* 🛡️ Immunity & Infection Defense
* 🍽️ Digestive Health & Gut Wellness
* 🌡️ Fever & Infectious Illnesses
* 👂 Ear Health & Hearing
* 🌿 General Health & Longevity

### 6. 📊 User Dashboard & History
* Secure SQLite authentication (`users.db`) with hashed passwords.
* Track recent predictions, confidence graphs, and historical quiz scores.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.11, Flask 3.0, Gunicorn, SQLite3
* **Machine Learning & NLP:** Scikit-learn, Sentence-Transformers, PyTorch, Joblib
* **Generative AI:** Google GenAI SDK (`gemini-3.5-flash-lite`)
* **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, FontAwesome, Google `<model-viewer>`
* **Deployment:** Docker, Render, Railway, Hugging Face Spaces

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/mukallaRajesh/Disease-Prediction-and-Hospital-Recommendation-System.git
cd Disease-Prediction-and-Hospital-Recommendation-System
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_secret_flask_key
```

### 5. Run the Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## ☁️ Deployment Guide

### Deploying to Render.com
1. Fork or push this repository to GitHub.
2. Log in to [Render](https://render.com) and click **New + $\to$ Web Service**.
3. Select this repository.
4. Render auto-detects `Procfile` and `.python-version`:
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn app:app`
5. In **Environment Variables**, add:
   * `GEMINI_API_KEY` = your Google Gemini API key.
6. Click **Deploy Web Service**.

---

## 📄 License
This project is open source and available under the [Apache 2.0 License](LICENSE).
