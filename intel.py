# Import necessary libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ระบุ path ให้ตรงกับตำแหน่งไฟล์
df = pd.read_csv('D:/solution.csv')

# ตรวจสอบข้อมูล
print(df.head())
# ตรวจสอบข้อมูลพื้นฐาน
print(df.info())
print(df.describe())

# เติมค่าที่หายไปด้วยค่าเฉลี่ย
df = df.fillna(df.mean())

# แบ่งข้อมูลเป็น X (คุณสมบัติ) และ y (เป้าหมาย)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดลต่างๆ
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVM': SVR(),
    'KNN': KNeighborsRegressor()
}

# ฝึกและประเมินผลโมเดลแต่ละตัว
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print("-" * 50)

# บันทึกโมเดล
joblib.dump(model, 'house_price_model.pkl')

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('house_price_model.pkl')

# เปรียบเทียบค่าจริงกับค่าทำนาย
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# แสดงกราฟ Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# แสดงกราฟ Residuals vs Predicted Values
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()
import pandas as pd

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('D:/solution.csv')

# ตรวจสอบข้อมูลแรก 5 แถว
print(df.head())

# ตรวจสอบข้อมูลเชิงลึก
print(df.info())  # แสดงข้อมูลประเภทของคอลัมน์
print(df.describe())  # สถิติพื้นฐานของข้อมูล
df = df.fillna(df.mean())
X = df.drop('SalePrice', axis=1)  # ลบคอลัมน์ 'SalePrice' เพื่อให้เหลือแค่คุณสมบัติ
y = df['SalePrice']  # ตัวแปรเป้าหมาย
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVM': SVR(),
    'KNN': KNeighborsRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)  # ฝึกโมเดล
    y_pred = model.predict(X_test)  # ทำนายค่าด้วยชุดทดสอบ
    
    mae = mean_absolute_error(y_test, y_pred)  # คำนวณ MAE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # คำนวณ RMSE
    r2 = r2_score(y_test, y_pred)  # คำนวณ R-squared
    
    print(f"Model: {name}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print("-" * 50)
import joblib

# สมมติว่าเราเลือกโมเดล Random Forest เป็นโมเดลที่ดีที่สุด
best_model = RandomForestRegressor()

# ฝึกโมเดลด้วยข้อมูลที่มี
best_model.fit(X_train, y_train)

# บันทึกโมเดล
joblib.dump(best_model, 'best_model.pkl')

# การโหลดโมเดลในอนาคต
model = joblib.load('best_model.pkl')
# สมมติว่าเราโหลดโมเดลที่บันทึกไว้
model = joblib.load('best_model.pkl')

# ใช้โมเดลในการทำนายผลลัพธ์จากข้อมูลใหม่
y_pred = model.predict(X_test)

# แสดงผลลัพธ์การทำนาย
print(y_pred)
