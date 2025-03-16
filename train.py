import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ดาวน์โหลดและโหลดชุดข้อมูล Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# แทนค่าที่เป็นศูนย์ในคอลัมน์ที่ไม่ควรมีค่าเป็นศูนย์ด้วย NaN
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# ใช้วิธีแทนค่าหายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์
imputer = SimpleImputer(strategy="mean")
df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])

# แยก Features และ Target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# แบ่งชุดข้อมูลเป็น train/test (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับค่าข้อมูลให้เป็นมาตรฐาน (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # สร้างและปรับมาตรฐานข้อมูลฝึก
X_test_scaled = scaler.transform(X_test)        # ปรับมาตรฐานข้อมูลทดสอบ

# สร้างโมเดล Neural Network
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # ชั้นแรก
model.add(Dense(32, activation='relu'))  # ชั้นที่สอง
model.add(Dense(1, activation='sigmoid'))  # ชั้น output
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ฝึกโมเดล
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=1)

