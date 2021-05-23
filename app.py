import streamlit as st
import os
import joblib

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

#Dictionaries
symboling_label = {0: 0, 1: 1, 2: 2, 3: 3, -2: 4, -1: 5}
make_label = {'plymouth': 0, 'porsche': 1, 'mazda': 2, 'audi': 3, 'dodge': 4, 'mercedes-benz': 5, 'saab': 6, 'bmw': 7, 'volkswagen': 8, 'mercury': 9, 'peugot': 10, 'honda': 11, 'subaru': 12, 'isuzu': 13, 'mitsubishi': 14, 'volvo': 15, 'alfa-romero': 16, 'chevrolet': 17, 'renault': 18, 'toyota': 19, 'jaguar': 20, 'nissan': 21}
fuel_type_label = {'gas': 0, 'diesel': 1}
aspiration_label = {'turbo': 0, 'std': 1}
num_of_doors_label = {'two': 0, 'four': 1}
body_style_label = {'hatchback': 0, 'convertible': 1, 'wagon': 2, 'sedan': 3, 'hardtop': 4}
drive_wheels_label = {'fwd': 0, 'rwd': 1, '4wd': 2}
engine_location_label = {'front': 0, 'rear': 1}
wheel_base_label = {102.4: 0, 102.9: 1, 104.5: 2, 104.3: 3, 97.0: 4, 95.1: 5, 100.4: 6, 97.3: 7, 107.9: 8, 108.0: 9, 95.7: 10, 109.1: 11, 96.6: 12, 96.1: 13, 95.3: 14, 96.9: 15, 93.1: 16, 93.0: 17, 86.6: 18, 96.3: 19, 88.6: 20, 88.4: 21, 89.5: 22, 91.3: 23, 93.3: 24, 93.7: 25, 94.3: 26, 94.5: 27, 96.5: 28, 96.0: 29, 95.9: 30, 99.8: 31, 99.4: 32, 99.5: 33, 101.2: 34, 103.5: 35, 103.3: 36, 105.8: 37, 102.0: 38, 98.8: 39, 104.9: 40, 106.7: 41, 110.0: 42, 102.7: 43, 112.0: 44, 113.0: 45, 97.2: 46, 115.6: 47, 99.2: 48, 114.2: 49, 98.4: 50, 99.1: 51, 120.9: 52}
height_label = {51.0: 0, 51.4: 1, 51.6: 2, 52.6: 3, 52.8: 4, 52.5: 5, 53.2: 6, 53.5: 7, 53.0: 8, 53.9: 9, 54.1: 10, 54.4: 11, 54.9: 12, 54.8: 13, 54.7: 14, 55.4: 15, 55.1: 16, 55.2: 17, 55.6: 18, 56.7: 19, 56.1: 20, 56.0: 21, 56.2: 22, 57.5: 23, 47.8: 24, 48.8: 25, 49.6: 26, 50.8: 27, 50.6: 28, 52.4: 29, 53.3: 30, 53.1: 31, 54.3: 32, 56.3: 33, 55.7: 34, 58.3: 35, 59.8: 36, 55.9: 37, 52.0: 38, 53.7: 39, 54.5: 40, 55.5: 41, 56.5: 42, 58.7: 43, 59.1: 44, 50.5: 45, 49.7: 46, 49.4: 47, 50.2: 48}
engine_type_label = {'ohc': 0, 'dohcv': 1, 'ohcf': 2, 'rotor': 3, 'dohc': 4, 'ohcv': 5, 'l': 6}
num_of_cylinders_label = {'eight': 0, 'six': 1, 'twelve': 2, 'three': 3, 'five': 4, 'two': 5, 'four': 6}
fuel_system_label = {'mpfi': 0, '2bbl': 1, 'spfi': 2, 'spdi': 3, '4bbl': 4, 'mfi': 5, '1bbl': 6, 'idi': 7}
bore_label = {2.68: 0, 3.19: 1, 3.5: 2, 3.47: 3, 3.13: 4, 3.31: 5, 3.62: 6, 2.91: 7, 3.03: 8, 2.97: 9, 3.34: 10, 2.92: 11, 2.99: 12, 3.08: 13, 3.76: 14, 3.58: 15, 3.17: 16, 3.59: 17, 3.33: 18, 3.74: 19, 3.24: 20, 3.01: 21, 3.6: 22, 3.43: 23, 3.78: 24, 3.35: 25, 3.61: 26, 3.94: 27, 3.27: 28, 2.54: 29, 3.63: 30, 3.54: 31, 3.46: 32, 3.8: 33, 3.7: 34, 3.05: 35, 3.15: 36, 3.39: 37}
stroke_label = {2.68: 0, 3.4: 1, 3.47: 2, 3.19: 3, 2.8: 4, 3.39: 5, 3.03: 6, 3.11: 7, 3.23: 8, 3.46: 9, 3.9: 10, 4.17: 11, 2.76: 12, 3.41: 13, 3.58: 14, 3.16: 15, 3.08: 16, 3.5: 17, 2.19: 18, 2.36: 19, 3.1: 20, 3.35: 21, 3.86: 22, 3.27: 23, 3.52: 24, 2.87: 25, 3.12: 26, 3.29: 27, 3.21: 28, 3.54: 29, 2.9: 30, 2.07: 31, 2.64: 32, 3.07: 33, 3.15: 34, 3.64: 35}
compression_ratio_label = {7.6: 0, 8.6: 1, 7.0: 2, 8.0: 3, 9.0: 4, 10.0: 5, 8.3: 6, 8.5: 7, 8.8: 8, 9.5: 9, 9.6: 10, 9.41: 11, 10.1: 12, 11.5: 13, 21.5: 14, 22.7: 15, 22.0: 16, 21.9: 17, 21.0: 18, 22.5: 19, 23.0: 20, 7.5: 21, 8.1: 22, 8.4: 23, 8.7: 24, 9.4: 25, 9.2: 26, 9.1: 27, 9.31: 28, 9.3: 29, 7.8: 30, 7.7: 31}
peak_rpm_label = {5250.0: 0, 5000.0: 1, 5900.0: 2, 4750.0: 3, 4500.0: 4, 5400.0: 5, 4250.0: 6, 4900.0: 7, 5800.0: 8, 4650.0: 9, 4400.0: 10, 5300.0: 11, 4150.0: 12, 4800.0: 13, 6600.0: 14, 5200.0: 15, 5600.0: 16, 4200.0: 17, 5100.0: 18, 6000.0: 19, 5750.0: 20, 5500.0: 21, 4350.0: 22}

@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

# Load model
def load_model(file_model):
    model = joblib.load(open(os.path.join(file_model), "rb"))
    return model

# Get value
def get_value(val, dict):
    for key, value in dict.items():
        if val == key:
            return value

# Get key
def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key

def fill_missing_val(df):
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

def main():
    """Car ML App"""

    st.title("Car Safety Evaluation Application")

    menu = ["Logistic Regression","Data Visualization"]
    choices = st.sidebar.selectbox("Select Activities", menu)
    if choices == 'Logistic Regression':
        st.subheader('Prediction')
        make = st.selectbox("Select Car Brand", tuple(make_label.keys()))
        fuel_type = st.selectbox("Select Fuel Type", tuple(fuel_type_label.keys()))
        aspiration = st.selectbox("Select Aspiration", tuple(aspiration_label.keys()))
        num_of_doors = st.selectbox("Select Number of Door", tuple(num_of_doors_label.keys()))
        body_style = st.selectbox("Select Body Style", tuple(body_style_label.keys()))
        drive_wheels = st.selectbox("Select Drive Wheels", tuple(drive_wheels_label.keys()))
        engine_location = st.selectbox("Select Engine Location", tuple(engine_location_label.keys()))
        wheel_base = st.selectbox("Select Wheel Base", tuple(wheel_base_label.keys()))
        #length = st.selectbox("Select Length of the Car", tuple(length_label.keys()))
        #width = st.selectbox("Select Width of the Car", tuple(width_label.keys()))
        height = st.selectbox("Select Height of the Car", tuple(height_label.keys()))
        #curb_weight = st.selectbox("Select Curb Weight of the Car", tuple(curb_weight_label.keys()))
        engine_type = st.selectbox("Select Type of Engine", tuple(engine_type_label.keys()))
        num_of_cylinders = st.selectbox("Select Number of Cylinders", tuple(num_of_cylinders_label.keys()))
        #engine_size = st.selectbox("Select Engine Size", tuple(engine_size_label.keys()))
        fuel_system = st.selectbox("Select Fuel System", tuple(fuel_system_label.keys()))
        bore = st.selectbox("Select Bore of the Car", tuple(bore_label.keys()))
        stroke = st.selectbox("Select Stroke of the Car", tuple(stroke_label.keys()))
        compression_ratio = st.selectbox("Select Compression Ratio", tuple(compression_ratio_label.keys()))
        #horsepower = st.selectbox("Select Horsepower", tuple(horsepower_label.keys()))
        peak_rpm = st.selectbox("Select Peak RPM", tuple(peak_rpm_label.keys()))

        value_make = get_value(make, make_label)
        value_fuel_type = get_value(fuel_type, fuel_type_label)
        value_aspiration = get_value(aspiration, aspiration_label)
        value_num_of_doors = get_value(num_of_doors, num_of_doors_label)
        value_body_style = get_value(body_style, body_style_label)
        value_drive_wheels = get_value(drive_wheels, drive_wheels_label)
        value_engine_location = get_value(engine_location, engine_location_label)
        value_wheel_base = get_value(wheel_base, wheel_base_label)
        #value_length = get_value(length, length_label)
        #value_width = get_value(width, width_label)
        value_height = get_value(height, height_label)
        #value_curb_weight = get_value(curb_weight, curb_weight_label)
        value_engine_type = get_value(engine_type, engine_type_label)
        value_num_of_cylinders = get_value(num_of_cylinders, num_of_cylinders_label)
        #value_engine_size = get_value(engine_size, engine_size_label)
        value_fuel_system = get_value(fuel_system, fuel_system_label)
        value_bore = get_value(bore, bore_label)
        value_stroke = get_value(stroke, stroke_label)
        value_compression_ratio = get_value(compression_ratio,compression_ratio_label)
        #value_horsepower = get_value(horsepower, horsepower_label)
        value_peak_rpm = get_value(peak_rpm, peak_rpm_label)

        data_encoded = [value_make, value_fuel_type, value_aspiration, value_num_of_doors,value_body_style,value_drive_wheels,value_engine_location,value_wheel_base,value_height, value_engine_type,value_num_of_cylinders,value_fuel_system,value_bore, value_stroke,value_compression_ratio,value_peak_rpm]
        clean_data = np.array(data_encoded).reshape(1,-1)

        st.subheader("Your selection")
        st.write("Make: ", make)
        st.write("Fuel type: ", fuel_type)
        st.write("Aspiration: ", aspiration)
        st.write("Number of door: ", num_of_doors)
        st.write("Body style: ", body_style)
        st.write("Drive wheels: ", drive_wheels)
        st.write("Engine location: ", engine_location)
        st.write("Wheel base: ", wheel_base)
        st.write("Height: ", height)
        st.write("Engine type: ", engine_type)
        st.write("Number of cylinders: ", num_of_cylinders)
        st.write("Fuel system: ", fuel_system)
        st.write("Bore: ", bore)
        st.write("Stroke: ", stroke)
        st.write("Compression ratio: ", compression_ratio)
        st.write("Peak RPM: ", peak_rpm)

        if st.button("Evaluate"):
            # Read files
            df = pd.read_csv('car_specifications.csv')

            # Fill missing value
            fill_missing_val(df)

            # Drop unecessary column
            df.drop(['city-mpg', 'curb-weight', 'engine-size', 'highway-mpg', 'horsepower', 'length', 'price', 'width', 'normalized-losses'], axis=1,inplace=True)

            # Encoding data
            lb = LabelEncoder()
            for i in df.columns:
               df[i]=lb.fit_transform(df[i])

            # Split data
            x = np.array(df.drop(["symboling"], 1))
            y = np.array(df["symboling"])
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            predictor = load_model("logit_model.pkl")
            prediction = predictor.predict(clean_data)
            score = predictor.score(x_test, y_test)
            data_decoded = get_key(prediction, symboling_label)

            if(data_decoded == -3):
                st.success("Safe Car! <-> 6 Stars") 
            if(data_decoded == -2):
                st.success("Pretty Safe! <-> 5 Stars")
            if(data_decoded == -1):
                st.success("Quite Safe! <-> 4 Stars")
            if(data_decoded == 0):
                st.warning("Acceptable car! <-> 3 Stars")
            if(data_decoded == 1):
                st.error("Quite Risky! <-> 2 Stars")
            if(data_decoded == 2):
                st.error("Risky! <-> 1 Star")
            if(data_decoded == 3):
                st.error("Critically Risky! <-> 0 Star")
            st.write(round(score*100, 2), " % Accuracy")

    if choices == 'Data Visualization':
        df = pd.read_csv('car_specifications.csv')
        st.subheader('Data Frame')
        st.dataframe(df.head(10))

        if st.checkbox("Show missing values:"):
            st.write(df.isna().sum())
            st.write('Shape: ', df.shape)
        if st.checkbox("Label value counts:"):
            st.write(df['symboling'].value_counts())
            st.write(df['symboling'].value_counts().plot(kind = 'bar'))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        if st.checkbox("Label Encoder"):
            lb = LabelEncoder()
            df1 = df
            for i in df1.columns:
                df1[i]=lb.fit_transform(df1[i])
            st.write(df1.head(10))

if __name__ == '__main__':
    main()