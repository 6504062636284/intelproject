import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #2C2F33;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #23272A;
    }
    [data-testid="stSidebar"] {
        background-color: #1A1A1A !important;
    }
    .stSidebar .stSelectbox label {
        color: white !important;
    }
    .stButton>button {
        background-color: black !important;
        color: white !important;
        font-size: 14px;
        border-radius: 12px;
        padding: 8px 16px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        border: 1px solid white;
    }
    .stButton>button:hover {
        background-color: #333 !important;
    }
    h1, h2, h3, h4, h5, h6, 
    .stMarkdown, .stTextInput label, 
    .stCheckbox label, .stNumberInput label, 
    .stSlider label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page 1: Machine Learning Overview
def page_1():
    st.title("ข้อมูลการซื้อขายรถยนต์")
    st.image("car_image.png", width=100)  # ใช้รูปภาพที่เกี่ยวกับรถยนต์
    st.write(
        "ข้อมูลการซื้อขายรถยนต์ที่รวบรวมมา "
        "โดยมีรายละเอียดเกี่ยวกับ **ยี่ห้อรถ ขนาดเครื่องยนต์ จำนวนห้องโดยสาร ประเภทน้ำมัน ข้อมูลเกี่ยวกับล้อ และการประเมินผลจากลูกค้า** "
        "โดยผมได้นำ data set มาจาก Kaggle [Car Price Dataset](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data?select=CarPrice_Assignment.csv) ")
    st.header("อัลกอริธึมที่ใช้")
    st.write(
        "- **Linear Regression:** วิเคราะห์ความสัมพันธ์เชิงเส้น\n"
        "- **Decision Tree:** ใช้เงื่อนไขในการแบ่งข้อมูล\n"
        "- **Random Forest:** รวมหลาย Decision Tree เพื่อเพิ่มความแม่นยำ\n"
        "- **SVM:** ใช้เส้น Hyperplane แบ่งข้อมูล\n"
        "- **KNN:** ใช้ข้อมูลรอบข้างที่คล้ายคลึงกันในการพยากรณ์"
    )
    
    # ขั้นตอนพัฒนาโมเดล Machine Learning
    st.header("ขั้นตอนพัฒนาโมเดล Machine Learning")
    steps = [
        ("ขั้นตอนที่ 1", "โหลดข้อมูลที่เราจะใช้ในการฝึกโมเดลจากไฟล์ CSV และแสดงตัวอย่างข้อมูล", "step1_image.png"),
        ("ขั้นตอนที่ 2", "ทำการแยกแบรนด์จากคอลัมน์ CarName และแปลงค่าข้อมูลเชิงหมวดหมู่เป็น One-Hot Encoding", "step2_image.png"),
        ("ขั้นตอนที่ 3", "แบ่งข้อมูลออกเป็นชุดฝึก (Training Set) และชุดทดสอบ (Testing Set)", "step3_image.png"),
        ("ขั้นตอนที่ 4", "ฝึกโมเดลโดยใช้หลายอัลกอริทึมเช่น Linear Regression, Random Forest, Decision Tree, SVR, KNN และประเมินผล", "step4_image.png"),
        ("ขั้นตอนที่ 5", "ใช้ GridSearchCV เพื่อปรับแต่งโมเดลที่ดีที่สุด", "step5_image.png"),
        ("ขั้นตอนที่ 6", "บันทึกโมเดลที่ดีที่สุดที่ได้จากการฝึกและการปรับแต่ง", "step6_image.png")
    ]

    for title, description, image in steps:
        st.write(description)
        st.image(image, caption=title, use_container_width=True)




# Page 2: Demo Machine Learning
model = joblib.load("car_price_model_best.pkl")
def demo_page():
    st.title("Car Price Prediction Demo")
    st.write("กรุณากรอกข้อมูลของรถเพื่อลองทำนายราคาของรถ")

    # Collect user inputs
    car_name = st.text_input("Car Name", value="bmw x4")  # Will extract brand from this
    symboling = st.number_input("Symboling (Risk Factor)", min_value=-2, max_value=3, value=0)  # Added missing feature
    fuel_type = st.selectbox("Fuel Type", ["gas", "diesel"])
    aspiration = st.selectbox("Aspiration", ["std", "turbo"])
    door_number = st.selectbox("Number of Doors", ["two", "four"])
    car_body = st.selectbox("Car Body", ["convertible", "hardtop", "hatchback", "sedan", "wagon"])
    drive_wheel = st.selectbox("Drive Wheel", ["4wd", "fwd", "rwd"])
    engine_location = st.selectbox("Engine Location", ["front", "rear"])
    wheelbase = st.number_input("Wheelbase", min_value=0.0, value=100.0)
    car_length = st.number_input("Car Length", min_value=0.0, value=170.0)
    car_width = st.number_input("Car Width", min_value=0.0, value=65.0)
    car_height = st.number_input("Car Height", min_value=0.0, value=50.0)
    curb_weight = st.number_input("Curb Weight", min_value=0, value=2500)
    engine_type = st.selectbox("Engine Type", ["dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"])
    cylinder_number = st.selectbox("Cylinder Number", ["two", "three", "four", "five", "six", "twelve"])
    engine_size = st.number_input("Engine Size", min_value=0, value=120)
    fuel_system = st.selectbox("Fuel System", ["1bbl", "2bbl", "4bbl", "idi", "mfi", "mpfi", "spdi", "spfi"])
    bore_ratio = st.number_input("Bore Ratio", min_value=0.0, value=3.0)
    stroke = st.number_input("Stroke", min_value=0.0, value=3.0)
    compression_ratio = st.number_input("Compression Ratio", min_value=0.0, value=9.0)
    horsepower = st.number_input("Horsepower", min_value=0, value=100)
    peakrpm = st.number_input("Peak RPM", min_value=0, value=5000)
    citympg = st.number_input("City MPG", min_value=0, value=25)
    highwaympg = st.number_input("Highway MPG", min_value=0, value=30)

    if st.button("Predict Price"):
        # Create a dictionary of input data
        input_dict = {
            'symboling': symboling,
            'fueltype': fuel_type,
            'aspiration': aspiration,
            'doornumber': door_number,
            'carbody': car_body,
            'drivewheel': drive_wheel,
            'enginelocation': engine_location,
            'wheelbase': wheelbase,
            'carlength': car_length,
            'carwidth': car_width,
            'carheight': car_height,
            'curbweight': curb_weight,
            'enginetype': engine_type,
            'cylindernumber': cylinder_number,
            'enginesize': engine_size,
            'fuelsystem': fuel_system,
            'boreratio': bore_ratio,
            'stroke': stroke,
            'compressionratio': compression_ratio,
            'horsepower': horsepower,
            'peakrpm': peakrpm,
            'citympg': citympg,
            'highwaympg': highwaympg
        }

        # Extract brand from car_name (assuming car_name format is "brand model")
        brand = car_name.split()[0].lower() if car_name else "unknown"
        input_dict['brand'] = brand

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical variables
        categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 
                               'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand']
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns)

        # Get the expected feature order from the model
        expected_features = model.feature_names_in_

        # Align input_df_encoded with expected features
        # Add missing columns with zeros
        for col in expected_features:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0

        # Reorder columns to match the model's expected order
        input_df_encoded = input_df_encoded[expected_features]

        # Predict
        price_prediction = model.predict(input_df_encoded)
        st.write(f"Predicted Car Price: ${price_prediction[0]:,.2f}")

# Page 3: Neural Network Explanation
def page_3():
    st.title("Neural Network - Explanation")
    st.write("""
    **โมเดล Neural Network สำหรับการทำนายโรคเบาหวาน**
    ในโปรเจกต์นี้ เราใช้ Neural Network เพื่อทำนายว่าใครมีแนวโน้มเป็นโรคเบาหวาน โดยใช้ข้อมูลทางการแพทย์หลายตัวแปร
    """)
    
    st.write("### ฟีเจอร์ที่ใช้ในข้อมูล:")
    st.write("- **Pregnancies**: จำนวนการตั้งครรภ์\n- **Glucose**: ระดับกลูโคสในเลือด\n- **BloodPressure**: ความดันโลหิต\n"
             "- **SkinThickness**: ความหนาของผิวหนัง\n- **Insulin**: ระดับอินซูลินในเลือด\n- **BMI**: ดัชนีมวลกาย\n"
             "- **DiabetesPedigree**: การสืบทอดทางพันธุกรรมของโรคเบาหวาน\n- **Age**: อายุของบุคคล")

    st.write("### วิธีการทำงานของ Neural Network:")
    st.write("- ประกอบด้วย **Input Layer**, **Hidden Layers**, และ **Output Layer**\n"
             "- ใช้ **backpropagation** เพื่อเรียนรู้และปรับ weights")

    st.subheader("1. เตรียมข้อมูล")
    st.write("โหลดข้อมูลและจัดการ missing values โดยใช้ **SimpleImputer** เติมค่าเฉลี่ย")
    st.code("""
    df = pd.read_csv(url, names=columns)
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    imputer = SimpleImputer(strategy="mean")
    df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])
    """, language="python")

    st.subheader("2. การปรับข้อมูลให้เป็นมาตรฐาน")
    st.write("ใช้ **StandardScaler** เพื่อปรับข้อมูลให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1")
    st.code("""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    """, language="python")

    st.subheader("3. การสร้างโมเดล Neural Network")
    st.write("ประกอบด้วย Input Layer, Hidden Layers และ Output Layer")
    st.code("""
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0)
    """, language="python")

    st.subheader("4. การทำนายผล")
    st.write("ทำนายผลโดยใช้ threshold 0.5 เพื่อแยก Diabetes/No Diabetes")
    st.code("""
    y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype("int32")
    """, language="python")

# Page 4: Neural Network Demo
def page_4():
    import streamlit as st
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    import pandas as pd

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
    df = pd.read_csv(url, names=columns)

    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    imputer = SimpleImputer(strategy="mean")
    df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0)

    def predict(input_data):
        return (model.predict(input_data) > 0.5).astype("int32")

    def get_user_input():
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
        bmi = st.number_input("BMI", min_value=0, max_value=60, value=25)
        diabetes_pedigree = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        return scaler.transform(input_data)

    def display_prediction(prediction):
        if prediction == 1:
            st.success("The model predicts: Diabetes")
        else:
            st.success("The model predicts: No Diabetes")

    st.title("Neural Network Model - Diabetes Prediction")
    st.write("This model uses Neural Network to predict diabetes.")
    input_data = get_user_input()
    if st.button("Predict"):
        prediction = predict(input_data)
        display_prediction(prediction)

# Main app navigation
page = st.sidebar.selectbox("Select a page", ["Machine Learning", "Demo Machine Learning", "Neural Network", "Demo Neural Network"])

if page == "Machine Learning":
    page_1()
elif page == "Demo Machine Learning":
    demo_page()
elif page == "Neural Network":
    page_3()
else:
    page_4()