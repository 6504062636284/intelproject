import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#โหลด Dataset 
df = pd.read_csv("D:/codepython/CarPrice_Assignment.csv")

# ดูตัวอย่างข้อมูล
print("ตัวอย่างข้อมูล:")
print(df.head())

# เตรียมข้อมูล (Preprocessing) 
df['brand'] = df['CarName'].apply(lambda x: x.split()[0])
df.drop('CarName', axis=1, inplace=True)

# ระบุคอลัมน์เชิงหมวดหมู่ที่ต้องการแปลง (object type)
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody',
                       'drivewheel', 'enginelocation', 'enginetype',
                       'cylindernumber', 'fuelsystem', 'brand']

# ใช้ One-Hot Encoding แปลงคอลัมน์เชิงหมวดหมู่โดย drop_first เพื่อลด multicollinearity
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# กำหนด features (X) และ target (y)
# target คือ 'price' และไม่ใช้ 'car_ID'
X = df_encoded.drop(columns=["car_ID", "price"])
y = df_encoded["price"]

#แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# พัฒนาโมเดลด้วยอัลกอริทึมที่แตกต่างกัน 

# กำหนด dictionary สำหรับเก็บโมเดล
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor()
}

results = {}

print("\nผลการประเมินโมเดล:")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    print(f"\nโมเดล: {name}")
    print(f"  MAE: {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R-squared: {r2:.4f}")

# ปรับแต่งโมเดลที่ดีที่สุด 
best_model_name = max(results, key=lambda x: results[x]["R2"])
print(f"\nโมเดลที่ดีที่สุด: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

# หากต้องการใช้การปรับแต่งด้วย GridSearchCV สำหรับ Random Forest:
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters from GridSearchCV:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# ประเมินผลโมเดลที่ได้จาก GridSearchCV
predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nผลประเมินโมเดลที่ปรับแต่ง:")
print(f"  MAE: {mae:,.2f}")
print(f"  RMSE: {rmse:,.2f}")
print(f"  R-squared: {r2:.4f}")

#บันทึกโมเดลที่ดีที่สุด 
joblib.dump(best_model, 'car_price_model_best.pkl')
print("บันทึกโมเดลเรียบร้อยแล้วในไฟล์ 'car_price_model_best.pkl'")
