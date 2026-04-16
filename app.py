import customtkinter as ctk
import pickle
import numpy as np
import pandas as pd

print("🚀 App starting...")

# Load model safely
try:
    with open("model/heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Heart Disease Risk Predictor")
app.geometry("500x700")

print("✅ UI window created")

# Title
ctk.CTkLabel(app, text="Heart Disease Predictor", font=("Arial", 20)).pack(pady=10)

# Inputs
age = ctk.CTkEntry(app, placeholder_text="Age")
age.pack(pady=5)

sex = ctk.CTkOptionMenu(app, values=["Male (1)", "Female (0)"])
sex.pack(pady=5)

cp = ctk.CTkOptionMenu(app, values=["0","1","2","3"])
cp.pack(pady=5)

trestbps = ctk.CTkEntry(app, placeholder_text="BP")
trestbps.pack(pady=5)

chol = ctk.CTkEntry(app, placeholder_text="Cholesterol")
chol.pack(pady=5)

fbs = ctk.CTkOptionMenu(app, values=["0","1"])
fbs.pack(pady=5)

restecg = ctk.CTkOptionMenu(app, values=["0","1","2"])
restecg.pack(pady=5)

thalach = ctk.CTkEntry(app, placeholder_text="Max HR")
thalach.pack(pady=5)

exang = ctk.CTkOptionMenu(app, values=["0","1"])
exang.pack(pady=5)

oldpeak = ctk.CTkEntry(app, placeholder_text="Oldpeak")
oldpeak.pack(pady=5)

slope = ctk.CTkOptionMenu(app, values=["0","1","2"])
slope.pack(pady=5)

ca = ctk.CTkOptionMenu(app, values=["0","1","2","3","4"])
ca.pack(pady=5)

thal = ctk.CTkOptionMenu(app, values=["0","1","2","3"])
thal.pack(pady=5)

# Output
result = ctk.CTkLabel(app, text="", font=("Arial", 16))
result.pack(pady=20)

# Predict
def predict():
    try:
        print("🔥 Button clicked")

        sex_val = 1 if "Male" in sex.get() else 0

        values = [
            int(age.get()), sex_val, int(cp.get()), int(trestbps.get()),
            int(chol.get()), int(fbs.get()), int(restecg.get()),
            int(thalach.get()), int(exang.get()), float(oldpeak.get()),
            int(slope.get()), int(ca.get()), int(thal.get())
        ]

        data = pd.DataFrame([values], columns=[
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"
        ])

        prob = model.predict_proba(data)[0][1] * 100

        print("Probability:", prob)

        if prob < 30:
            result.configure(text=f"🟢 Low Risk {prob:.2f}%", text_color="green")
        elif prob < 70:
            result.configure(text=f"🟡 Medium Risk {prob:.2f}%", text_color="yellow")
        else:
            result.configure(text=f"🔴 High Risk {prob:.2f}%", text_color="red")

    except Exception as e:
        print("❌ ERROR:", e)
        result.configure(text="Error", text_color="red")

# Button
ctk.CTkButton(app, text="Predict", command=predict).pack(pady=10)

print("✅ Starting UI loop...")

app.mainloop()