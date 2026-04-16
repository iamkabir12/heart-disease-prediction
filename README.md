# ❤️ Heart Disease Risk Predictor

A **Machine Learning-based application** that predicts the **risk of heart disease** using clinical parameters.
This project combines a trained ML model with a modern GUI for real-time predictions.

---

## 🚀 Features

* 🔍 Predicts heart disease risk with **percentage**
* 🧠 Uses **Random Forest Classifier (~91% accuracy)**
* 🖥️ Modern GUI using **CustomTkinter**
* 🎯 Color-coded risk output:

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk
* ⚡ Fast and user-friendly interface

---

## 🧠 Model Details

* **Algorithm:** Random Forest Classifier
* **Accuracy:** ~91% (Cross-validation)
* **Dataset:** UCI Heart Disease Dataset
* **Important Features:**

  * Chest Pain Type (`cp`)
  * Number of Vessels (`ca`)
  * Thalassemia (`thal`)
  * Max Heart Rate (`thalach`)
  * Exercise Induced Angina (`exang`)

---

## 📸 Preview

![App UI](screenshot.png)

---

## 🖥️ How to Run

```bash
# Clone repository
git clone https://github.com/iamkabir12/heart-disease-prediction.git

# Navigate into folder
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## 📂 Project Structure

```
heart-disease-prediction/
│
├── data/                # Dataset (heart.csv)
├── model/               # Trained model (heart_model.pkl)
├── app.py               # GUI application
├── train_model.py       # Model training script
├── requirements.txt     # Dependencies
├── README.md            # Documentation
├── screenshot.png       # App preview image
```

---

## 📌 Tech Stack

* Python
* Scikit-learn
* Pandas & NumPy
* CustomTkinter
* Matplotlib & Seaborn

---

## 🎯 Future Improvements

* 🌐 Deploy as a web application
* 📊 Add graphical risk visualization
* 💡 Provide personalized health recommendations
* 📈 Improve model with more data

---

## 👨‍💻 Author

**Pratik Pandey (KABIR)**
GitHub: https://github.com/iamkabir12

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
