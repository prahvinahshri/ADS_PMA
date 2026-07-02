import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/wessamsw/Airline_Passenger_Satisfaction/main/airline_passenger_satisfaction.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

st.title("✈️ Airline Passenger Satisfaction Dashboard")
st.markdown("### Customer Experience Analytics — Agile Data Science PMA")

# --- Interactive Feature 1: Sidebar Dropdowns ---
st.sidebar.header("🔍 Filter Options")

travel_class = st.sidebar.selectbox(
    "Select Travel Class",
    options=["All"] + sorted(df['Class'].unique().tolist())
)

travel_type = st.sidebar.selectbox(
    "Select Type of Travel",
    options=["All"] + sorted(df['Type of Travel'].unique().tolist())
)

# --- Interactive Feature 2: Age Slider ---
age_range = st.sidebar.slider(
    "Select Age Range",
    int(df['Age'].min()),
    int(df['Age'].max()),
    (20, 60)
)

# Apply filters
filtered_df = df.copy()
if travel_class != "All":
    filtered_df = filtered_df[filtered_df['Class'] == travel_class]
if travel_type != "All":
    filtered_df = filtered_df[filtered_df['Type of Travel'] == travel_type]
filtered_df = filtered_df[
    (filtered_df['Age'] >= age_range[0]) &
    (filtered_df['Age'] <= age_range[1])
]

st.markdown(f"**Showing {len(filtered_df):,} passengers**")

# --- Visualization 1: Satisfaction Distribution ---
st.subheader("1. Passenger Satisfaction Distribution")
fig1, ax1 = plt.subplots(figsize=(6, 4))
filtered_df['Satisfaction'].value_counts().plot(
    kind='bar', color=['#e74c3c', '#2ecc71'], ax=ax1)
ax1.set_title("Satisfaction Count")
ax1.set_xlabel("Satisfaction")
ax1.set_ylabel("Count")
ax1.tick_params(axis='x', rotation=0)
plt.tight_layout()
st.pyplot(fig1)

# --- Visualization 2: Age Distribution ---
st.subheader("2. Age Distribution of Passengers")
fig2, ax2 = plt.subplots(figsize=(6, 4))
filtered_df['Age'].hist(bins=30, color='steelblue',
                        edgecolor='black', ax=ax2)
ax2.set_title("Passenger Age Distribution")
ax2.set_xlabel("Age")
ax2.set_ylabel("Frequency")
plt.tight_layout()
st.pyplot(fig2)

# --- Visualization 3: Average Service Ratings ---
st.subheader("3. Average In-flight Service Ratings")
service_cols = ['Seat Comfort', 'Food and Drink', 'In-flight Service',
                'In-flight Entertainment', 'Cleanliness', 'Leg Room Service']
avg_ratings = filtered_df[service_cols].mean().sort_values()
fig3, ax3 = plt.subplots(figsize=(8, 4))
avg_ratings.plot(kind='barh', color='steelblue', ax=ax3)
ax3.set_title("Average Service Quality Ratings")
ax3.set_xlabel("Average Rating (1-5)")
plt.tight_layout()
st.pyplot(fig3)

# --- Predictive Output ---
st.subheader("4. 🔮 Predict Passenger Satisfaction")
st.markdown("Fill in passenger details to get a satisfaction prediction:")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=35)
    flight_distance = st.number_input("Flight Distance (km)",
                                      min_value=0, max_value=10000, value=1000)
    seat_comfort = st.slider("Seat Comfort Rating", 1, 5, 3)
    inflight_wifi = st.slider("In-flight Wifi Rating", 1, 5, 3)
    online_boarding = st.slider("Online Boarding Rating", 1, 5, 3)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    travel_class_pred = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    travel_type_pred = st.selectbox("Type of Travel",
                                    ["Business travel", "Personal Travel"])
    inflight_entertainment = st.slider("In-flight Entertainment Rating", 1, 5, 3)
    food_drink = st.slider("Food and Drink Rating", 1, 5, 3)

if st.button("🚀 Predict Satisfaction"):
    gender_enc = 1 if gender == "Male" else 0
    class_map = {"Business": 0, "Eco": 1, "Eco Plus": 2}
    class_enc = class_map[travel_class_pred]
    travel_enc = 0 if travel_type_pred == "Business travel" else 1

    from sklearn.preprocessing import StandardScaler
    temp_scaler = StandardScaler()
    temp_scaler.fit(df[['Age', 'Flight Distance',
                         'Departure Delay', 'Arrival Delay']].fillna(0))
    scaled_vals = temp_scaler.transform([[age, flight_distance, 0, 0]])[0]

    input_data = pd.DataFrame([[
        gender_enc, 1, travel_enc, class_enc,
        scaled_vals[0], scaled_vals[1], 0, 0,
        3, 3, 3, online_boarding, 3, 3,
        seat_comfort, 3, 3, food_drink, 3,
        inflight_wifi, inflight_entertainment, 3
    ]], columns=[
        'Gender_encoded', 'Customer Type_encoded',
        'Type of Travel_encoded', 'Class_encoded',
        'Age_scaled', 'Flight Distance_scaled',
        'Departure Delay_scaled', 'Arrival Delay_scaled',
        'Departure and Arrival Time Convenience',
        'Ease of Online Booking', 'Check-in Service',
        'Online Boarding', 'Gate Location', 'On-board Service',
        'Seat Comfort', 'Leg Room Service', 'Cleanliness',
        'Food and Drink', 'In-flight Service',
        'In-flight Wifi Service', 'In-flight Entertainment',
        'Baggage Handling'
    ])

    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        if prediction == 1:
            st.success(
                f"✅ Predicted: SATISFIED "
                f"(Confidence: {probability[1]*100:.1f}%)")
        else:
            st.error(
                f"❌ Predicted: NEUTRAL OR DISSATISFIED "
                f"(Confidence: {probability[0]*100:.1f}%)")
    except:
        st.warning("Model file not found.")

# =============================================
# Q5 — MONITORING SECTION
# =============================================
st.markdown("---")
st.header("📊 5. Monitoring Metrics & Drift Analysis")

# --- Monitoring Metric 1: Satisfaction Rate ---
st.subheader("Monitoring Metric 1: Overall Satisfaction Rate")
satisfaction_rate = (df['Satisfaction'] == 'Satisfied').mean() * 100
filtered_sat_rate = (filtered_df['Satisfaction'] == 'Satisfied').mean() * 100

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="Overall Satisfaction Rate",
        value=f"{satisfaction_rate:.2f}%",
        help="Percentage of all passengers who are satisfied"
    )
with col2:
    st.metric(
        label="Filtered Satisfaction Rate",
        value=f"{filtered_sat_rate:.2f}%",
        delta=f"{filtered_sat_rate - satisfaction_rate:.2f}% vs overall",
        help="Satisfaction rate for currently filtered passengers"
    )

# --- Monitoring Metric 2: Data Quality ---
st.subheader("Monitoring Metric 2: Data Quality")
missing_total = df.isnull().sum().sum()
duplicate_total = df.duplicated().sum()
invalid_cleanliness = len(df[(df['Cleanliness'] < 1) | (df['Cleanliness'] > 5)])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Missing Values", value=missing_total)
with col2:
    st.metric(label="Duplicate Rows", value=duplicate_total)
with col3:
    st.metric(label="Invalid Cleanliness Ratings", value=invalid_cleanliness)

# --- Drift Analysis ---
st.subheader("Data Drift Analysis: Old vs New Data")
st.markdown("*Simulating drift by comparing first half (old) vs second half (new) of dataset*")

mid = len(df) // 2
old_data = df.iloc[:mid]
new_data = df.iloc[mid:]

old_sat = (old_data['Satisfaction'] == 'Satisfied').mean() * 100
new_sat = (new_data['Satisfaction'] == 'Satisfied').mean() * 100
old_delay = old_data['Departure Delay'].mean()
new_delay = new_data['Departure Delay'].mean()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Satisfaction Rate Drift**")
    drift_data = pd.DataFrame({
        'Period': ['Old Data (First Half)', 'New Data (Second Half)'],
        'Satisfaction Rate (%)': [round(old_sat, 2), round(new_sat, 2)]
    })
    fig_drift1, ax_drift1 = plt.subplots(figsize=(5, 3))
    ax_drift1.bar(drift_data['Period'], drift_data['Satisfaction Rate (%)'],
                  color=['#3498db', '#e67e22'])
    ax_drift1.set_title("Satisfaction Rate: Old vs New")
    ax_drift1.set_ylabel("Satisfaction Rate (%)")
    ax_drift1.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig_drift1)
    st.metric("Satisfaction Drift",
              f"{abs(new_sat - old_sat):.2f}%",
              delta=f"{new_sat - old_sat:.2f}%")

with col2:
    st.markdown("**Departure Delay Drift**")
    delay_data = pd.DataFrame({
        'Period': ['Old Data (First Half)', 'New Data (Second Half)'],
        'Avg Delay (mins)': [round(old_delay, 2), round(new_delay, 2)]
    })
    fig_drift2, ax_drift2 = plt.subplots(figsize=(5, 3))
    ax_drift2.bar(delay_data['Period'], delay_data['Avg Delay (mins)'],
                  color=['#3498db', '#e67e22'])
    ax_drift2.set_title("Avg Departure Delay: Old vs New")
    ax_drift2.set_ylabel("Average Delay (minutes)")
    plt.tight_layout()
    st.pyplot(fig_drift2)
    st.metric("Delay Drift",
              f"{abs(new_delay - old_delay):.2f} mins",
              delta=f"{new_delay - old_delay:.2f} mins")
