
# 🚖 FareCast AI — Intelligent Ride Fare Prediction

**FareCast AI** is a machine learning web application that estimates ride fares for trips across Delhi NCR using trip details such as vehicle type, ride distance, pickup location, drop-off location, and travel time.

🌐 **Live Demo:** https://farecast-ai.onrender.com  
📘 **API Documentation:** https://farecast-ai.onrender.com/docs  
💻 **GitHub Repository:** https://github.com/SuryanshKumarSingh1113/FareCast-AI

---

## 📌 Problem Statement

Ride fares can vary depending on trip characteristics and other factors. FareCast AI explores how machine learning can estimate a ride's fare from historical booking data.

The project demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to API development and cloud deployment.

## ✨ Key Features

- 🚗 Fare estimation based on selected vehicle type
- 📍 Pickup and drop-off location inputs
- 📏 Optional ride distance input
- 🕒 Trip time and date features
- ⚡ Interactive web interface
- 🔌 REST API built with FastAPI
- ☁️ Cloud deployment on Render
- 🧠 Trained machine learning pipeline saved with Joblib

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Models | Linear Regression, Random Forest Regressor |
| Model Serialization | Joblib |
| API | FastAPI |
| API Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |
| Version Control | Git and GitHub |

## 📊 Dataset

The project uses the **Uber Data India — NCR Ride Bookings** dataset obtained from Kaggle.

- Approximately 150,000 booking records
- 21 original columns
- Completed rides are filtered for model training
- Records with missing or invalid target fares and distances are excluded

### Data Preprocessing

1. Loaded and inspected the raw dataset.
2. Filtered completed bookings.
3. Removed records with missing or invalid fare and distance values.
4. Extracted hour, day of the week, and month from trip date and time.
5. Imputed missing feature values where required.
6. Encoded categorical features using One-Hot Encoding.
7. Split the data into training and testing sets.

**Input features:** Vehicle Type, Ride Distance, Pickup Location, Drop Location, Hour, Day of Week, and Month.

**Target variable:** Booking Value.

## 🤖 Model Development

Two regression algorithms were evaluated:

- Linear Regression
- Random Forest Regressor

A median-fare baseline was also evaluated for comparison. The model pipeline with the lower Root Mean Squared Error (RMSE) was selected.

### Evaluation Results

The recorded evaluation results were:

| Model | MAE (₹) | RMSE (₹) | R² Score |
|---|---:|---:|---:|
| Median Baseline | 274.59 | 409.57 | -0.0516 |
| Linear Regression | 281.44 | 392.42 | 0.0346 |
| Random Forest | 282.11 | 393.46 | 0.0295 |

**Interpretation:** Linear Regression achieved the lowest RMSE among these evaluated models, while the median baseline achieved the lowest MAE. The low R² scores indicate that the current features explain only a small portion of fare variation.

These results highlight the importance of feature engineering, data quality, and further model improvement.

## 🏗️ Project Structure

```text
FareCast-AI/
├── app.py
├── train_model.py
├── inspect_data.py
├── farecast_model.joblib
├── requirements.txt
├── render.yaml
├── .gitignore
├── .python-version
└── static/
    └── index.html
```

The dataset files are excluded from the Git repository. Obtain the dataset separately and place it in the `data/` directory before running the inspection or training scripts.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SuryanshKumarSingh1113/FareCast-AI.git
cd FareCast-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
uvicorn app:app --reload
```

Open these URLs in your browser:

- Web application: http://127.0.0.1:8000/
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

The saved model file must be available for the application to load successfully.

## 🔌 API Usage

The application exposes a prediction endpoint at `POST /predict`.

Example request:

```json
{
  "vehicle_type": "Auto",
  "ride_distance": 12.5,
  "pickup_location": "Khandsa",
  "drop_location": "Cyber Hub",
  "hour": 10,
  "day_of_week": 2,
  "month": 10
}
```

The API returns a predicted fare based on the trained model.

Use the interactive documentation at `/docs` to test the endpoint.

## ☁️ Deployment

FareCast AI is deployed as a web service on Render.

- **Live application:** https://farecast-ai.onrender.com
- **API documentation:** https://farecast-ai.onrender.com/docs

The trained model is loaded by the API, allowing users to request fare predictions through the web interface.

## ⚠️ Limitations

- Prediction quality is limited by the current dataset and available features.
- The model estimates fares from historical Uber NCR booking data.
- It does not retrieve live prices from Uber, Ola, or Rapido.
- Actual ride prices may differ from predicted values.
- Predictions may be less reliable for unusual trips or data patterns not represented in the training data.

## 🔮 Future Enhancements

- Improve predictive performance through feature engineering and model tuning.
- Evaluate additional regression algorithms.
- Add stronger validation and monitoring.
- Incorporate more reliable distance, traffic, and time-related features where available.
- Explore live fare comparison through authorized APIs, subject to availability.

## 🎯 Project Objective

This project demonstrates practical skills in data preprocessing, regression modeling, model evaluation, API development, model serialization, and cloud deployment.

---

**Developed by Suryansh Kumar Singh**  
*Data Science Engineering | Machine Learning Project*
