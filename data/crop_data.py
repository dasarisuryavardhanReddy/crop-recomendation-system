"""
OptiCrop Dataset Generator
Generates a realistic crop recommendation dataset based on agricultural research data.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Crop-specific parameter ranges based on agricultural research
crop_params = {
    'rice':        {'N': (60,100), 'P': (30,60),  'K': (30,60),  'temp': (20,30), 'humidity': (80,95), 'ph': (5.5,7.0), 'rainfall': (150,300)},
    'maize':       {'N': (60,100), 'P': (50,80),  'K': (50,80),  'temp': (18,28), 'humidity': (55,75), 'ph': (5.5,7.5), 'rainfall': (50,120)},
    'chickpea':    {'N': (20,50),  'P': (50,80),  'K': (60,100), 'temp': (15,25), 'humidity': (15,40), 'ph': (6.0,8.0), 'rainfall': (30,100)},
    'kidneybeans': {'N': (20,50),  'P': (50,80),  'K': (15,40),  'temp': (15,25), 'humidity': (60,80), 'ph': (5.5,7.0), 'rainfall': (40,100)},
    'pigeonpeas':  {'N': (15,40),  'P': (50,80),  'K': (15,40),  'temp': (20,30), 'humidity': (40,70), 'ph': (5.0,7.0), 'rainfall': (50,150)},
    'mothbeans':   {'N': (15,40),  'P': (30,60),  'K': (15,40),  'temp': (28,38), 'humidity': (25,55), 'ph': (6.0,8.0), 'rainfall': (30,80)},
    'mungbean':    {'N': (15,40),  'P': (30,60),  'K': (15,40),  'temp': (25,35), 'humidity': (50,80), 'ph': (6.0,7.5), 'rainfall': (30,90)},
    'blackgram':   {'N': (15,40),  'P': (40,70),  'K': (15,40),  'temp': (25,35), 'humidity': (60,80), 'ph': (6.0,7.5), 'rainfall': (50,120)},
    'lentil':      {'N': (15,40),  'P': (50,80),  'K': (15,40),  'temp': (10,20), 'humidity': (40,60), 'ph': (6.0,8.0), 'rainfall': (25,65)},
    'pomegranate': {'N': (15,40),  'P': (10,30),  'K': (30,60),  'temp': (20,30), 'humidity': (40,60), 'ph': (5.5,7.5), 'rainfall': (50,100)},
    'banana':      {'N': (80,120), 'P': (50,80),  'K': (200,250),'temp': (22,32), 'humidity': (75,90), 'ph': (5.5,7.0), 'rainfall': (100,200)},
    'mango':       {'N': (15,40),  'P': (10,30),  'K': (30,60),  'temp': (24,35), 'humidity': (50,80), 'ph': (5.5,7.5), 'rainfall': (50,150)},
    'grapes':      {'N': (15,40),  'P': (10,30),  'K': (30,60),  'temp': (15,25), 'humidity': (50,80), 'ph': (5.5,7.0), 'rainfall': (30,80)},
    'watermelon':  {'N': (60,100), 'P': (10,30),  'K': (40,80),  'temp': (25,35), 'humidity': (65,90), 'ph': (5.5,7.0), 'rainfall': (40,100)},
    'muskmelon':   {'N': (80,120), 'P': (10,30),  'K': (40,80),  'temp': (25,35), 'humidity': (65,90), 'ph': (6.0,7.5), 'rainfall': (20,60)},
    'apple':       {'N': (15,40),  'P': (10,30),  'K': (150,200),'temp': (5,15),  'humidity': (50,80), 'ph': (5.5,7.0), 'rainfall': (80,120)},
    'orange':      {'N': (15,40),  'P': (10,30),  'K': (10,30),  'temp': (20,30), 'humidity': (70,90), 'ph': (6.0,7.5), 'rainfall': (80,150)},
    'papaya':      {'N': (40,80),  'P': (10,30),  'K': (40,80),  'temp': (25,35), 'humidity': (75,90), 'ph': (6.0,7.5), 'rainfall': (100,200)},
    'coconut':     {'N': (15,40),  'P': (10,30),  'K': (15,40),  'temp': (25,35), 'humidity': (80,95), 'ph': (5.0,8.0), 'rainfall': (100,200)},
    'cotton':      {'N': (100,140),'P': (40,70),  'K': (15,40),  'temp': (25,35), 'humidity': (40,70), 'ph': (6.0,8.0), 'rainfall': (60,120)},
    'jute':        {'N': (60,100), 'P': (40,70),  'K': (30,60),  'temp': (25,35), 'humidity': (70,90), 'ph': (6.0,7.5), 'rainfall': (150,250)},
    'coffee':      {'N': (80,120), 'P': (10,30),  'K': (15,40),  'temp': (18,28), 'humidity': (50,80), 'ph': (6.0,7.5), 'rainfall': (150,250)},
}

samples_per_crop = 100
records = []

for crop, params in crop_params.items():
    for _ in range(samples_per_crop):
        record = {
            'N':        np.random.uniform(*params['N']),
            'P':        np.random.uniform(*params['P']),
            'K':        np.random.uniform(*params['K']),
            'temperature': np.random.uniform(*params['temp']),
            'humidity': np.random.uniform(*params['humidity']),
            'ph':       np.random.uniform(*params['ph']),
            'rainfall': np.random.uniform(*params['rainfall']),
            'label':    crop
        }
        records.append(record)

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = os.path.join(os.path.dirname(__file__), 'Crop_recommendation.csv')
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")
print(f"Shape: {df.shape}")
print(f"Crops: {df['label'].unique()}")
print(df.describe())
