import os
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image, ImageFile
from openai import OpenAI   # <-- modern client

# --------- FIX TRUNCATED IMAGES ----------
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------- BASIC SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------- LOAD YOUR LEAF MODEL ----------
classifier = pipeline(
    "image-classification",
    model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    use_fast=True
)

# ======== OPENROUTER SETUP =========
OPENROUTER_KEY = "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"

client = OpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1"
)
# ====================================

# --------- SINGLE SOLUTION FUNCTION ----------
def get_treatment_from_gpt(disease):

    prompt = f"""
    Give ONLY 5 simple, practical farmer-level solutions
    to cure this plant disease: {disease}.
    Return exactly 5 bullet points.
    """

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": "You are an agricultural doctor."},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content

        # Convert to clean list
        lines = [l.strip("•- ") for l in text.split("\n") if l.strip()]
        return lines[:5]

    except Exception as e:
        print("⚠️ OpenRouter failed:", e)

        # ---- SAFE FALLBACK (app will NEVER crash) ----
        return [
            f"Remove severely infected parts of {disease}.",
            "Avoid over-watering the plant.",
            "Ensure good sunlight and airflow.",
            "Use a general fungicide if needed.",
            "Keep tools clean to avoid spreading infection."
        ]

# --------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")

    result = classifier(img)
    disease = result[0]["label"]
    confidence = round(result[0]["score"] * 100, 2)

    treatment = get_treatment_from_gpt(disease)

    return jsonify({
        "image": "/static/uploads/" + file.filename,
        "disease": disease,
        "confidence": confidence,
        "treatment": treatment
    })

if __name__ == "__main__":
    app.run(debug=True)
