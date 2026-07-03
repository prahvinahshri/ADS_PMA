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

# --- Custom CSS ---
st.markdown("""
<style>
    /* Background image — airplane wing/sky */
    .stApp {
        background-image: 
            linear-gradient(
                rgba(235, 245, 255, 0.92),
                rgba(235, 245, 255, 0.92)
            ),
            url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 50, 120, 0.92) !important;
        border-right: 3px solid #1A73E8;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #A8D4FF !important;
        font-weight: bold;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.85);
        border: 2px solid #1A73E8;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 2px 4px 12px rgba(26, 115, 232, 0.2);
    }

    /* Main title */
    h1 {
        color: #0A3278 !important;
        font-family: Arial, sans-serif !important;
        text-align: center;
    }

    h2 {
        color: #1A73E8 !important;
        font-family: Arial, sans-serif !important;
        border-left: 4px solid #1A73E8;
        padding-left: 10px;
    }

    h3 {
        color: #0A3278 !important;
        font-family: Arial, sans-serif !important;
    }

    /* Button */
    .stButton > button {
        background-color: #1A73E8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 2px 2px 8px rgba(26, 115, 232, 0.3);
    }

    .stButton > button:hover {
        background-color: #0A3278;
        box-shadow: 2px 4px 12px rgba(26, 115, 232, 0.5);
    }

    /* Info box */
    .info-box {
        background-color: rgba(26, 115, 232, 0.1);
        border-left: 4px solid #1A73E8;
        border-radius: 5px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #0A3278;
        font-size: 0.95em;
    }

    /* Chart containers */
    .chart-container {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 2px 4px 12px rgba(26, 115, 232, 0.15);
        margin-bottom: 20px;
    }

    /* Section divider */
    hr {
        border: none;
        border-top: 2px solid #1A73E8;
        margin: 25px 0;
        opacity: 0.3;
    }

    /* General text */
    p, li {
        color: #0A3278;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib style to match light theme
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = '#F0F7FF'
mpl.rcParams['axes.edgecolor'] = '#1A73E8'
mpl.rcParams['text.color'] = '#0A3278'
mpl.rcParams['axes.labelcolor'] = '#0A3278'
mpl.rcParams['xtick.color'] = '#0A3278'
mpl.rcParams['ytick.color'] = '#0A3278'
mpl.rcParams['grid.color'] = '#BDD7F5'
mpl.rcParams['grid.alpha'] = 0.5

DATA_URL = "https://raw.githubusercontent.com/wessamsw/Airline_Passenger_Satisfaction/main/airline_passenger_satisfaction.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# ─── HEADER ─────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; padding: 20px 0 10px 0;'>
    <h1 style='font-size: 2.8em; color: #0A3278; 
               text-shadow: 1px 1px 3px rgba(0,0,0,0.1);'>
        ✈️ Airline Passenger Satisfaction Dashboard
    </h1>
    <p style='color: #1A73E8; font-size: 1.15em; font-weight: bold;'>
        Customer Experience Analytics — Agile Data Science PMA
    </p>
    <p style='color: #555; font-size: 0.95em;'>
        Powered by Random Forest Classification Model | MRTB 2173
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── SIDEBAR ────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align: center; padding: 15px 0 5px 0;'>
    <h2 style='color: white; font-size: 1.3em;'>🔍 Filter Options</h2>
    <p style='color: #A8D4FF; font-size: 0.85em;'>
        Adjust filters to explore passenger data
    </p>
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

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center;'>
    <p style='color: #A8D4FF; font-size: 0.8em;'>
        MRTB 2173 Agile Data Science PMA<br>
        Airline Passenger Satisfaction
    </p>
</div>
""", unsafe_allow_html=True)

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

# ─── SUMMARY METRICS ────────────────────────────────────
st.markdown("## 📋 Summary Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("✈️ Total Passengers", f"{len(filtered_df):,}")
with col2:
    sat_rate = (filtered_df['Satisfaction'] == 'Satisfied').mean() * 100
    st.metric("😊 Satisfaction Rate", f"{sat_rate:.1f}%")
with col3:
    avg_delay = filtered_df['Departure Delay'].mean()
    st.metric("⏱️ Avg Departure Delay", f"{avg_delay:.1f} mins")
with col4:
    avg_distance = filtered_df['Flight Distance'].mean()
    st.metric("🗺️ Avg Flight Distance", f"{avg_distance:.0f} km")

st.markdown("---")

# ─── VISUALIZATIONS ─────────────────────────────────────
st.markdown("## 📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Passenger Satisfaction Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    counts = filtered_df['Satisfaction'].value_counts()
    colors_bar = ['#E74C3C', '#27AE60']
    bars = ax1.bar(counts.index, counts.values,
                   color=colors_bar,
                   edgecolor='white',
                   linewidth=1.5,
                   width=0.5)
    ax1.set_title("Satisfaction Count",
                  color='#0A3278', fontsize=13, fontweight='bold')
    ax1.set_xlabel("Satisfaction", color='#0A3278')
    ax1.set_ylabel("Count", color='#0A3278')
    ax1.tick_params(axis='x', rotation=0)
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 200,
                 f'{count:,}',
                 ha='center', va='bottom',
                 color='#0A3278', fontweight='bold', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.subheader("2. Age Distribution of Passengers")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    filtered_df['Age'].hist(bins=30,
                             color='#1A73E8',
                             edgecolor='white',
                             linewidth=0.8,
                             ax=ax2,
                             alpha=0.85)
    ax2.set_title("Passenger Age Distribution",
                  color='#0A3278', fontsize=13, fontweight='bold')
    ax2.set_xlabel("Age", color='#0A3278')
    ax2.set_ylabel("Frequency", color='#0A3278')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.subheader("3. Average In-flight Service Ratings")
service_cols = ['Seat Comfort', 'Food and Drink', 'In-flight Service',
                'In-flight Entertainment', 'Cleanliness',
                'Leg Room Service']
avg_ratings = filtered_df[service_cols].mean().sort_values()
mean_rating = avg_ratings.mean()
colors_service = ['#1A73E8' if v >= mean_rating
                  else '#64B5F6' for v in avg_ratings.values]

fig3, ax3 = plt.subplots(figsize=(10, 4))
bars3 = ax3.barh(avg_ratings.index,
                  avg_ratings.values,
                  color=colors_service,
                  edgecolor='white',
                  linewidth=0.8,
                  alpha=0.9)
ax3.set_title("Average Service Quality Ratings (1-5 scale)",
              color='#0A3278', fontsize=13, fontweight='bold')
ax3.set_xlabel("Average Rating (1-5)", color='#0A3278')
ax3.set_xlim(0, 5.5)
for bar, val in zip(bars3, avg_ratings.values):
    ax3.text(val + 0.05,
             bar.get_y() + bar.get_height()/2.,
             f'{val:.2f}',
             va='center',
             color='#0A3278',
             fontweight='bold',
             fontsize=10)
ax3.axvline(x=mean_rating,
             color='#E74C3C',
             linestyle='--',
             alpha=0.8,
             linewidth=2,
             label=f'Average ({mean_rating:.2f})')
ax3.legend(facecolor='white', edgecolor='#1A73E8')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig3, use_container_width=True)

st.markdown("---")

# ─── PREDICTIVE OUTPUT ──────────────────────────────────
st.markdown("## 🔮 Predict Passenger Satisfaction")
st.markdown("""
<div class='info-box'>
    Fill in passenger details below to generate a real-time satisfaction 
    prediction using the trained Random Forest model (96.39% accuracy).
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**✈️ Flight & Passenger Details**")
    age = st.number_input("Age", min_value=1, max_value=100, value=35)
    flight_distance = st.number_input("Flight Distance (km)",
                                      min_value=0, max_value=10000,
                                      value=1000)
    gender = st.selectbox("Gender", ["Male", "Female"])
    travel_class_pred = st.selectbox(
        "Class", ["Business", "Eco", "Eco Plus"])
    travel_type_pred = st.selectbox(
        "Type of Travel", ["Business travel", "Personal Travel"])

with col2:
    st.markdown("**⭐ Service Quality Ratings (1=Poor, 5=Excellent)**")
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
                         'Departure Delay',
                         'Arrival Delay']].fillna(0))
    scaled_vals = temp_scaler.transform(
        [[age, flight_distance, 0, 0]])[0]

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
            st.metric("😊 Satisfaction Probability",
                      f"{probability[1]*100:.1f}%")
        with col2:
            st.metric("😞 Dissatisfaction Probability",
                      f"{probability[0]*100:.1f}%")
    except:
        st.warning("⚠️ Model file not found. "
                   "Ensure best_model.pkl exists in the project root.")

st.markdown("---")

# ─── MONITORING SECTION ─────────────────────────────────
st.markdown("## 📈 Monitoring Metrics & Drift Analysis")

st.subheader("Monitoring Metric 1: Overall Satisfaction Rate")
satisfaction_rate = (df['Satisfaction'] == 'Satisfied').mean() * 100
filtered_sat_rate = (
    filtered_df['Satisfaction'] == 'Satisfied').mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Overall Satisfaction Rate",
              f"{satisfaction_rate:.2f}%")
with col2:
    st.metric("Filtered Satisfaction Rate",
              f"{filtered_sat_rate:.2f}%",
              delta=f"{filtered_sat_rate - satisfaction_rate:.2f}% vs overall")
with col3:
    st.metric("Passengers in Current View",
              f"{len(filtered_df):,}")

st.subheader("Monitoring Metric 2: Data Quality")
missing_total = df.isnull().sum().sum()
duplicate_total = df.duplicated().sum()
invalid_cleanliness = len(
    df[(df['Cleanliness'] < 1) | (df['Cleanliness'] > 5)])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Missing Values", missing_total,
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
    Simulating drift by comparing first half (old data) vs 
    second half (new data) of the dataset to detect shifts 
    in passenger behaviour and operational patterns over time.
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
              color=['#1A73E8', '#FF7043'],
              edgecolor='white',
              linewidth=1.5,
              width=0.4)
    ax_d1.set_title("Satisfaction Rate: Old vs New",
                    color='#0A3278', fontsize=11, fontweight='bold')
    ax_d1.set_ylabel("Satisfaction Rate (%)", color='#0A3278')
    ax_d1.set_ylim(0, 100)
    for i, v in enumerate([old_sat, new_sat]):
        ax_d1.text(i, v + 1, f'{v:.2f}%',
                   ha='center', color='#0A3278',
                   fontweight='bold', fontsize=11)
    ax_d1.spines['top'].set_visible(False)
    ax_d1.spines['right'].set_visible(False)
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
              color=['#1A73E8', '#FF7043'],
              edgecolor='white',
              linewidth=1.5,
              width=0.4)
    ax_d2.set_title("Avg Departure Delay: Old vs New",
                    color='#0A3278', fontsize=11, fontweight='bold')
    ax_d2.set_ylabel("Average Delay (minutes)", color='#0A3278')
    for i, v in enumerate([old_delay, new_delay]):
        ax_d2.text(i, v + 0.2, f'{v:.2f}',
                   ha='center', color='#0A3278',
                   fontweight='bold', fontsize=11)
    ax_d2.spines['top'].set_visible(False)
    ax_d2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_d2, use_container_width=True)
    st.metric("Delay Drift",
              f"{abs(new_delay - old_delay):.2f} mins",
              delta=f"{new_delay - old_delay:.2f} mins")

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 15px 0;
            color: #1A73E8; font-size: 0.9em;'>
    ✈️ MRTB 2173 Agile Data Science PMA | 
    Airline Passenger Satisfaction Analysis |
    Random Forest Model (96.39% Accuracy)
</div>
""", unsafe_allow_html=True)
