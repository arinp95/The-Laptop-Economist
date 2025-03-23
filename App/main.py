import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

st.markdown(
    """
    <div style="display: flex; justify-content: flex-end;">
        <a href="https://github.com/arinp95/Laptop-Price-Predictor" target="_blank">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub"/>
        </a>
        <a href="https://www.linkedin.com/in/arinp95/" target="_blank" style="margin-left: 10px;">
            <img src="https://img.icons8.com/ios-filled/30/000000/linkedin.png" alt="LinkedIn"/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Import model and data
st.title("Laptop Price Predictor 💻")
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pd.read_parquet('df.parquet', engine='pyarrow')

# Define placeholders
placeholders = {
    "company": "Select a brand",
    "laptop_type": "Select a type",
    "ram": "Select RAM",
    "touchscreen": "Select option",
    "ips": "Select option",
    "resolution": "Select resolution",
    "cpu": "Select CPU",
    "hdd": "Select HDD",
    "ssd": "Select SSD",
    "gpu": "Select GPU",
    "os": "Select OS"
}

# Define columns for layout
left_column, middle_column, right_column = st.columns(3)
with left_column:
    company = st.selectbox("Brand", ["Select a brand"] + list(df["Company"].unique()), key="company")

with middle_column:
    laptop_type = st.selectbox("Type", ["Select a type"] + list(df["TypeName"].unique()), key="laptop_type")

with right_column:
    ram = st.selectbox("Ram (in GB)", ["Select RAM"] + sorted(df["Ram"].unique()), key="ram")

left_column, middle_column, right_column = st.columns(3)
with left_column:
    weight = st.number_input("Weight of laptop in kg", min_value=1.0, value=1.0, step=0.5)

with middle_column:
    touchscreen = st.selectbox("Touchscreen", ["Select option", "No", "Yes"], key="touchscreen")

with right_column:
    ips = st.selectbox("IPS Display", ["Select option", "No", "Yes"], key="ips")

left_column, middle_column, right_column = st.columns(3)
with left_column:
    Screen_size = st.number_input("Screen Size (in Inches)", min_value=13.5, value=13.5, step=0.1)

with middle_column:
    resolution = st.selectbox('Screen Resolution',
                              ["Select resolution"] + ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                       '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
                              key="resolution")

with right_column:
    cpu = st.selectbox("CPU Processor", ["Select CPU"] + list(df["CPU Processor"].unique()), key="cpu")

left_column, right_column = st.columns(2)
with left_column:
    hdd = st.selectbox("HDD(in GB)", ["Select HDD"] + [0, 128, 256, 512, 1024, 2048], key="hdd")

with right_column:
    ssd = st.selectbox("SSD(in GB)", ["Select SSD"] + [128, 256, 512, 1024], key="ssd")

gpu = st.selectbox("GPU Brand", ["Select GPU"] + list(df["GPU brand"].unique()), key="gpu")
os = st.selectbox("OS Type", ["Select OS"] + list(df["OS"].unique()), key="os")

# Define a function to display error messages
def display_error(message):
    st.error(message)

if st.button("Predict Price"):
    errors = []

    if company == placeholders["company"]:
        errors.append("Brand")
    if laptop_type == placeholders["laptop_type"]:
        errors.append("Type")
    if ram == placeholders["ram"]:
        errors.append("RAM")
    if touchscreen == placeholders["touchscreen"]:
        errors.append("Touchscreen")
    if ips == placeholders["ips"]:
        errors.append("IPS Display")
    if resolution == placeholders["resolution"]:
        errors.append("Screen Resolution")
    if cpu == placeholders["cpu"]:
        errors.append("CPU Processor")
    if hdd == placeholders["hdd"]:
        errors.append("HDD")
    if ssd == placeholders["ssd"]:
        errors.append("SSD")
    if gpu == placeholders["gpu"]:
        errors.append("GPU Brand")
    if os == placeholders["os"]:
        errors.append("OS Type")

    if errors:
        display_error(f"Please select all fields before predicting the price.")
        display_error(f"Missing Fields: {', '.join(errors)}.")
    else:
        ppi = None
        touchscreen = 1 if touchscreen == "Yes" else 0
        ips = 1 if ips == "Yes" else 0

        X_res = int(resolution.split("x")[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / Screen_size

        query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        try:
            prediction = pipe.predict(query)
            predicted_price = int(np.exp(prediction[0]))
            formatted_price = "{:,}".format(predicted_price)
            st.title(f"The Predicted Price of Laptop is Rs {formatted_price}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
