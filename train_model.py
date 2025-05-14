import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from category_encoders import TargetEncoder

# ------------------ Load and Clean Data ------------------
df = pd.read_csv("Bengaluru_House_Data.csv")
df.dropna(inplace=True)
df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else np.nan)

def convert_sqft_to_num(x):
    try:
        if isinstance(x, str):
            if '-' in x:
                tokens = x.split('-')
                return (float(tokens[0]) + float(tokens[1])) / 2
            match = re.findall(r"\d+\.?\d*", x)
            if match:
                return float(match[0])
        return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df.dropna(subset=['total_sqft', 'bhk'], inplace=True)
df = df[(df['total_sqft'] / df['bhk']) >= 250]
df = df[df['bath'] < df['bhk'] + 3]
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
df = df[df['price_per_sqft'] < df['price_per_sqft'].quantile(0.98)]

df['total_rooms'] = df['bhk'] + df['bath'] + df['balcony']
df['sqft_per_bhk'] = df['total_sqft'] / df['bhk']

location_counts = df['location'].value_counts()
rare_locations = location_counts[location_counts < 10].index
df['location'] = df['location'].apply(lambda x: 'Other' if x in rare_locations else x)

# ------------------ Prepare Features ------------------
X = df[['total_sqft', 'bath', 'balcony', 'bhk', 'location', 'total_rooms', 'sqft_per_bhk']]
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['location']
numerical_features = ['total_sqft', 'bath', 'balcony', 'bhk', 'total_rooms', 'sqft_per_bhk']

preprocessor = ColumnTransformer([
    ('cat', TargetEncoder(), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

X_train_transformed = preprocessor.fit_transform(X_train, y_train)

# ------------------ Best Models ------------------
rf_best = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=2, min_samples_leaf=1, random_state=42)
gb_best = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
xgb_best = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)

rf_best.fit(X_train_transformed, y_train)
gb_best.fit(X_train_transformed, y_train)
xgb_best.fit(X_train_transformed, y_train)

stacked_model = StackingRegressor(
    estimators=[('rf', rf_best), ('gb', gb_best), ('xgb', xgb_best)],
    final_estimator=Ridge(),
    passthrough=True,
    n_jobs=-1
)

# ------------------ Final Pipeline ------------------
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', stacked_model)
])

model_pipeline.fit(X_train, y_train)

# ------------------ Evaluate ------------------
y_pred_log = model_pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)
score = r2_score(y_test_actual, y_pred)

print(f"✅ Model trained successfully. R² Score: {score * 100:.2f}%")

# ------------------ Save model ------------------
joblib.dump(model_pipeline, "model.pkl")
print("✅ Model saved as model.pkl")
