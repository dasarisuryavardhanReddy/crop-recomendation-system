# OptiCrop — Smart Agricultural Production Optimization Engine

An AI-powered crop recommendation system using machine learning and Flask.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data/crop_data.py

# Train models
python train_model.py

# Run the app
python app.py
```

Then open http://localhost:5000 in your browser.

## 📁 Project Structure

```
OptiCrop/
├── app.py                    # Flask application (main entry point)
├── train_model.py            # ML model training pipeline
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── crop_data.py          # Dataset generator
│   └── Crop_recommendation.csv
│
├── model/
│   ├── model.pkl             # Trained Random Forest model
│   ├── scaler.pkl            # StandardScaler
│   ├── encoder.pkl           # LabelEncoder
│   └── model_summary.json    # Model performance metrics
│
├── static/
│   ├── css/style.css         # Premium dark UI design system
│   └── images/plots/         # Generated analytics charts
│
└── templates/
    ├── index.html            # Home + Prediction Form
    ├── result.html           # Prediction Result Page
    ├── analytics.html        # ML Analytics Dashboard
    ├── crops.html            # Crop Catalog
    └── about.html            # About / Documentation
```

## 🤖 Machine Learning Models

| Model               | Test Accuracy | CV Mean  |
|---------------------|--------------|----------|
| K-Nearest Neighbors | 89.8%        | 90.2%    |
| Logistic Regression | 92.5%        | 90.9%    |
| Decision Tree       | 89.8%        | 92.3%    |
| **Random Forest** ✅  | **93.9%**    | **94.1%** |

## 🌾 Supported Crops (22)

Cereals, Legumes, Fruits, Cash Crops including:
Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean,
Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon,
Musk Melon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

## 📊 Input Parameters

| Parameter   | Unit  | Range   | Description            |
|-------------|-------|---------|------------------------|
| Nitrogen    | kg/ha | 0-200   | Soil nitrogen content  |
| Phosphorous | kg/ha | 0-150   | Soil phosphorous level |
| Potassium   | kg/ha | 0-300   | Soil potassium level   |
| Temperature | °C    | -10-50  | Average temperature    |
| Humidity    | %     | 0-100   | Relative humidity      |
| pH          | -     | 0-14    | Soil acidity/alkalinity|
| Rainfall    | mm    | 0-500   | Annual rainfall        |

## 🔌 API Usage

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N":82,"P":45,"K":43,"temperature":25,"humidity":88,"ph":6.5,"rainfall":220}'
```

## 📄 License

For educational and research purposes. OptiCrop Smart Agricultural Production Optimization Engine.
