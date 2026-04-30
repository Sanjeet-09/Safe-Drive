import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf

# --- CONFIG & ASSET PATHS ---
MODEL_FILE = 'road_accident_model.h5'
METADATA_FILE = 'road_accident_assets.joblib'

st.set_page_config(page_title="SafeDrive AI", layout="wide")

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0f1117; }
    
    .hero-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #16213e 50%, #0f3460 100%);
        border-radius: 15px; padding: 40px; margin-bottom: 30px; border: 1px solid #ffffff11;
        text-align: center;
    }
    
    .section-card { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 12px; padding: 25px; margin-bottom: 20px; }
    .section-title { font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 15px; }

    .card { border-radius: 12px; padding: 25px; text-align: center; }
    .danger { background: linear-gradient(135deg, #7f1d1d, #b91c1c); color: white; border: 1px solid #ef4444; }
    .safe { background: linear-gradient(135deg, #14532d, #15803d); color: white; border: 1px solid #22c55e; }
    
    .advice-container { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_everything():
    if not os.path.exists(METADATA_FILE) or not os.path.exists(MODEL_FILE): return None
    bundle = joblib.load(METADATA_FILE)
    bundle['model'] = tf.keras.models.load_model(MODEL_FILE)
    return bundle

assets = load_everything()

# Header
st.markdown('<div class="hero-box"><h1 style="color:white; margin:0;">SafeDrive AI</h1>'
            '<p style="color:#94a3b8; margin:0;">Neural Network Road Safety Analysis</p></div>', unsafe_allow_html=True)

if not assets:
    st.error("Missing model files! Please run your training notebook first.")
    st.stop()

# --- MAIN INPUT SECTION ---
st.markdown('<div class="section-card"><div class="section-title">Environmental Conditions</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: road_type = st.selectbox("Road Type", ["rural", "highway", "urban"])
with c2: weather = st.selectbox("Weather", ["clear", "foggy", "rainy"])
with c3: lighting = st.selectbox("Lighting", ["daylight", "dim", "night"])
with c4: time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">Infrastructure & Speed</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 2])
with c1: lanes = st.number_input("Lanes", 1, 10, 2)
with c2: speed = st.slider("Target Speed (km/h)", 20, 120, 60)
with c3: curve = st.slider("Road Curvature", 0.0, 1.0, 0.2)
st.markdown('</div>', unsafe_allow_html=True)

# Analysis Trigger
if st.button("Generate Risk Assessment", use_container_width=True):
    # Data Prep
    df = pd.DataFrame([[road_type, lighting, weather, time_of_day, lanes, curve, speed]], 
                     columns=assets['categorical_columns'] + assets['numerical_columns'])
    df[assets['numerical_columns']] = assets['scaler'].transform(df[assets['numerical_columns']])
    encoded = pd.get_dummies(df)
    
    final_features = pd.DataFrame(0, index=[0], columns=assets['feature_columns'])
    for col in encoded.columns:
        if col in final_features.columns: final_features[col] = encoded[col].values[0]

    # Prediction
    risk_score = assets['model'].predict(final_features, verbose=0)[0][0]
    high_risk = risk_score > 0.5
    
    # Results Display
    res_col1, res_col2 = st.columns([1, 1.5])
    with res_col1:
        card_type = "danger" if high_risk else "safe"
        # Removed the <p> tag text from here
        st.markdown(f'<div class="card {card_type}"><h2>{"HIGH RISK" if high_risk else "SAFE"}</h2></div>', unsafe_allow_html=True)

    with res_col2:
        st.write(f"#### AI Risk Level: {risk_score*100:.1f}%")
        st.progress(float(risk_score))
        # Advice section is now completely removed

        
        # st.markdown('<div class="advice-container"><b>Actionable Advice:</b>', unsafe_allow_html=True)
        # if high_risk: st.warning("Significant risk detected. Consider reducing your speed.")
        # if weather != "clear": st.info(f"Visibility is impacted by {weather} weather.")
        # if not high_risk: st.success("Statistically safe, stay alert!")
        # st.markdown('</div>', unsafe_allow_html=True)
