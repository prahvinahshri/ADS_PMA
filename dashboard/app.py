import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import numpy as np

st.set_page_config(
    page_title="Airline Satisfaction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E2130;
        border-right: 2px solid #4472C4;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1E2130;
        border: 1px solid #4472C4;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 8px rgba(68, 114, 196, 0.3);
    }
    
    /* Headers */
    h1 {
        color: #4472C4 !important;
        font-family: Arial, sans-serif !important;
        border-bottom: 2px solid #4472C4;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #A9C4F5 !important;
        font-family: Arial, sans-serif !important;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #A9C4F5 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4472C4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2E5AAC;
        color: white;
    }
    
    /* Selectbox and slider labels */
    .stSelectbox label, .stSlider label {
        color: #A9C4F5 !important;
        font-weight: bold;
    }
    
    /* Divider */
    hr {
        border-color: #4472C4;
        margin: 20px 0;
    }

    /* Charts fill width */
    .stPlotlyChart, .stImage {
        width: 100% !important;
    }

    /* Info boxes */
    .info-box {
        background-color: #1E2130;
        border-left: 4px solid #4472C4;
        border-radius: 5px;
        padding: 10px 15px;
        margin: 10px 0;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib dark style to match dashboard
plt.style.use('dark_background')
mpl.rcParams['figure.facecolor'] = '#1E2130'
mpl.rcParams['axes.facecolor'] = '#1E2130'
mpl.rcParams['axes.edgecolor'] = '#4472C4'
mpl.rcParams['text.color'] = '#FAFAFA'
mpl.rcParams['axes.labelcolor'] = '#FAFAFA'
mpl.rcParams['xtick.color'] = '#FAFAFA'
mpl.rcParams['ytick.color'] = '#FAFAFA'
mpl.rcParams['grid.color'] = '#2E3250'
mpl.rcParams['grid.alpha'] = 0.5

DATA_URL = "https://raw.githubusercontent.com/wessamsw/Airline_Passenger_Satisfaction/main/airline_passenger_satisfaction.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# ─── HEADER ─────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='font-size: 2.5em; color: #4472C4;'>
        ✈️ Airline Passenger Satisfaction Dashboard
    </h1>
    <p style='color: #A9C4F5; font-size: 1.1em;'>
        Customer Experience Analytics — Agile Data Science PMA
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── SIDEBAR ────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h2 style='color: #4472C4;'>🔍 Filter Options</h2>
</div>
""", unsafe_allow_html=True)

travel_class = st.sidebar.selectbox(
    "Select Travel Class",
    options=["All"] + sorted(df['Class'].unique().tolist())
)

travel_type = st.sidebar.selectbox(
    "Select Type of Travel",
    options=["All"] + sorted(df['Type of Travel'].unique().tolist())
)

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

# Summary metrics at top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Passengers", f"{len(filtered_df):,}")
with col2:
    sat_rate = (filtered_df['Satisfaction'] == 'Satisfied').mean() * 100
    st.metric("Satisfaction Rate", f"{sat_rate:.1f}%")
with col3:
    avg_delay = filtered_df['Departure Delay'].mean()
    st.metric("Avg Departure Delay", f"{avg_delay:.1f} mins")
with col4:
    avg_distance = filtered_df['Flight Distance'].mean()
    st.metric("Avg Flight Distance", f"{avg_distance:.0f} km")

st.markdown("---")

# ─── VISUALIZATIONS ─────────────────────────────────────
st.markdown("## 📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Passenger Satisfaction Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    counts = filtered_df['Satisfaction'].value_counts()
    bars = ax1.bar(counts.index, counts.values,
                   color=['#E74C3C', '#2ECC71'],
                   edgecolor='#4472C4', linewidth=1.5,
                   width=0.5)
    ax1.set_title("Satisfaction Count", color='#A9C4F5', fontsize=13)
    ax1.set_xlabel("Satisfaction", color='#FAFAFA')
    ax1.set_ylabel("Count", color='#FAFAFA')
    ax1.tick_params(axis='x', rotation=0)
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'{count:,}', ha='center', va='bottom',
                color='#FAFAFA', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.subheader("2. Age Distribution of Passengers")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    filtered_df['Age'].hist(bins=30, color='#4472C4',
                             edgecolor='#A9C4F5',
                             linewidth=0.8, ax=ax2, alpha=0.85)
    ax2.set_title("Passenger Age Distribution",
                  color='#A9C4F5', fontsize=13)
    ax2.set_xlabel("Age", color='#FAFAFA')
    ax2.set_ylabel("Frequency", color='#FAFAFA')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.subheader("3. Average In-flight Service Ratings")
service_cols = ['Seat Comfort', 'Food and Drink', 'In-flight Service',
                'In-flight Entertainment', 'Cleanliness', 'Leg Room Service']
avg_ratings = filtered_df[service_cols].mean().sort_values()
colors = ['#4472C4' if v < avg_ratings.mean()
          else '#2ECC71' for v in avg_ratings.values]

fig3, ax3 = plt.subplots(figsize=(10, 4))
bars3 = ax3.barh(avg_ratings.index, avg_ratings.values,
                  color=colors, edgecolor='#A9C4F5',
                  linewidth=0.8, alpha=0.85)
ax3.set_title("Average Service Quality Ratings (1-5 scale)",
              color='#A9C4F5', fontsize=13)
ax3.set_xlabel("Average Rating (1-5)", color='#FAFAFA')
ax3.set_xlim(0, 5)
for bar, val in zip(bars3, avg_ratings.values):
    ax3.text(val + 0.05, bar.get_y() + bar.get_height()/2.,
             f'{val:.2f}', va='center', color='#FAFAFA',
             fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.axvline(x=avg_ratings.mean(), color='#E74C3C',
             linestyle='--', alpha=0.7, label='Average')
ax3.legend(facecolor='#1E2130', edgecolor='#4472C4')
plt.tight_layout()
st.pyplot(fig3, use_container_width=True)

st.markdown("---")

# ─── PREDICTIVE OUTPUT ──────────────────────────────────
st.markdown("## 🔮 Predict Passenger Satisfaction")
st.markdown("""
<div class='info-box'>
Fill in passenger details below to generate a satisfaction prediction
using the trained Random Forest model.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**✈️ Flight Details**")
    age = st.number_input("Age", min_value=1, max_value=100, value=35)
    flight_distance = st.number_input("Flight Distance (km)",
                                      min_value=0, max_value=10000, value=1000)
    gender = st.selectbox("Gender", ["Male", "Female"])
    travel_class_pred = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    travel_type_pred = st.selectbox("Type of Travel",
                                    ["Business travel", "Personal Travel"])

with col2:
    st.markdown("**⭐ Service Ratings (1-5)**")
    seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
    inflight_wifi = st.slider("In-flight Wifi", 1, 5, 3)
    online_boarding = st.slider("Online Boarding", 1, 5, 3)
    inflight_entertainment = st.slider("In-flight Entertainment", 1, 5, 3)
    food_drink = st.slider("Food and Drink", 1, 5, 3)

st.markdown("")
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

        st.markdown("")
        if prediction == 1:
            st.success(
                f"✅ Predicted: SATISFIED  |  "
                f"Confidence: {probability[1]*100:.1f}%")
        else:
            st.error(
                f"❌ Predicted: NEUTRAL OR DISSATISFIED  |  "
                f"Confidence: {probability[0]*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Satisfaction Probability",
                      f"{probability[1]*100:.1f}%")
        with col2:
            st.metric("Dissatisfaction Probability",
                      f"{probability[0]*100:.1f}%")
    except:
        st.warning("⚠️ Model file not found. "
                   "Please ensure best_model.pkl exists.")

st.markdown("---")

# ─── MONITORING SECTION ─────────────────────────────────
st.markdown("## 📈 Monitoring Metrics & Drift Analysis")

st.subheader("Monitoring Metric 1: Overall Satisfaction Rate")
satisfaction_rate = (df['Satisfaction'] == 'Satisfied').mean() * 100
filtered_sat_rate = (filtered_df['Satisfaction'] == 'Satisfied').mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Overall Satisfaction Rate",
              f"{satisfaction_rate:.2f}%")
with col2:
    st.metric("Filtered Satisfaction Rate",
              f"{filtered_sat_rate:.2f}%",
              delta=f"{filtered_sat_rate - satisfaction_rate:.2f}% vs overall")
with col3:
    st.metric("Total Passengers in View",
              f"{len(filtered_df):,}")

st.subheader("Monitoring Metric 2: Data Quality")
missing_total = df.isnull().sum().sum()
duplicate_total = df.duplicated().sum()
invalid_cleanliness = len(
    df[(df['Cleanliness'] < 1) | (df['Cleanliness'] > 5)])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Missing Values",
              missing_total,
              delta="Arrival Delay column",
              delta_color="inverse")
with col2:
    st.metric("Duplicate Rows", duplicate_total)
with col3:
    st.metric("Invalid Cleanliness Ratings",
              invalid_cleanliness,
              delta_color="inverse")

st.markdown("---")
st.subheader("Data Drift Analysis: Old vs New Data")
st.markdown("""
<div class='info-box'>
Simulating drift by comparing first half (old) vs second half (new) of dataset
</div>
""", unsafe_allow_html=True)

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
    fig_d1, ax_d1 = plt.subplots(figsize=(5, 3))
    ax_d1.bar(['Old Data\n(First Half)', 'New Data\n(Second Half)'],
              [old_sat, new_sat],
              color=['#3498DB', '#E67E22'],
              edgecolor='#A9C4F5', linewidth=1, width=0.4)
    ax_d1.set_title("Satisfaction Rate: Old vs New",
                    color='#A9C4F5', fontsize=11)
    ax_d1.set_ylabel("Satisfaction Rate (%)", color='#FAFAFA')
    ax_d1.set_ylim(0, 100)
    for i, v in enumerate([old_sat, new_sat]):
        ax_d1.text(i, v + 1, f'{v:.2f}%', ha='center',
                  color='#FAFAFA', fontweight='bold')
    ax_d1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_d1, use_container_width=True)
    st.metric("Satisfaction Drift",
              f"{abs(new_sat - old_sat):.2f}%",
              delta=f"{new_sat - old_sat:.2f}%")

with col2:
    st.markdown("**Departure Delay Drift**")
    fig_d2, ax_d2 = plt.subplots(figsize=(5, 3))
    ax_d2.bar(['Old Data\n(First Half)', 'New Data\n(Second Half)'],
              [old_delay, new_delay],
              color=['#3498DB', '#E67E22'],
              edgecolor='#A9C4F5', linewidth=1, width=0.4)
    ax_d2.set_title("Avg Departure Delay: Old vs New",
                    color='#A9C4F5', fontsize=11)
    ax_d2.set_ylabel("Average Delay (minutes)", color='#FAFAFA')
    for i, v in enumerate([old_delay, new_delay]):
        ax_d2.text(i, v + 0.2, f'{v:.2f}', ha='center',
                  color='#FAFAFA', fontweight='bold')
    ax_d2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_d2, use_container_width=True)
    st.metric("Delay Drift",
              f"{abs(new_delay - old_delay):.2f} mins",
              delta=f"{new_delay - old_delay:.2f} mins")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F7F7F; font-size: 0.9em;'>
    MRTB 2173 Agile Data Science PMA | Airline Passenger Satisfaction Analysis
</div>
""", unsafe_allow_html=True)
