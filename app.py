import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="AI Predictor: Hip to GTK V3.2", layout="wide")

# ==========================================
# 2. Core Feature Engineering & Cleaning
# ==========================================
def engineer_features(df):
    """Extract 4 core columns, force numeric conversion, and calculate optical ratios"""
    mapping = {
        'lv-R': ['lv-R', 'RValue', 'R_Value', 'R-Value', 'LightRed_lv', 'lv-LightRed', 'lv_R'],
        'lv-G': ['lv-G', 'GValue', 'G_Value', 'G-Value', 'LightGreen_lv', 'lv-LightGreen', 'lv_G'],
        'lv-B': ['lv-B', 'BValue', 'B_Value', 'B-Value', 'LightBlue_lv', 'lv-LightBlue', 'lv_B'],
        'lv-W': ['lv-W', 'WValue', 'W_Value', 'W-Value', 'LightWhite_lv', 'lv-LightWhite', 'lv_W']
    }
    
    clean_df = pd.DataFrame(index=df.index)
    
    for standard_name, candidates in mapping.items():
        found_col = next((c for c in candidates if c in df.columns), None)
        if found_col:
            clean_df[standard_name] = pd.to_numeric(df[found_col], errors='coerce')
        else:
            raise ValueError(f"❌ Missing required brightness column: {standard_name}")

    # Drop dirty data (e.g., rows containing text like "Upper Limit")
    valid_mask = clean_df.notna().all(axis=1)
    clean_df = clean_df[valid_mask].copy()
    
    if len(clean_df) == 0:
        raise ValueError("❌ No valid numeric data found after cleaning!")

    X = clean_df.copy()
    w_safe = X['lv-W'].replace(0, 1e-5)
    X['r_ratio'] = X['lv-R'] / w_safe
    X['g_ratio'] = X['lv-G'] / w_safe
    X['b_ratio'] = X['lv-B'] / w_safe
    X['rgb_sum'] = X['lv-R'] + X['lv-G'] + X['lv-B']
    
    return X, valid_mask

# ==========================================
# 3. Dynamic Model Training
# ==========================================
@st.cache_resource
def load_and_train_model(filepath, _last_modified_time):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        targets = ['mix_W_2000_r_ratio_current', 'mix_W_2000_g_ratio_current', 
                   'mix_W_2000_b_ratio_current', 'w_4000_current']
        
        missing_targets = [t for t in targets if t not in df.columns]
        if missing_targets:
            raise ValueError(f"Training data is missing target columns: {missing_targets}")
            
        df = df.dropna(subset=targets).reset_index(drop=True)
        
        X_train, _ = engineer_features(df)
        y_train = df[targets]
        
        base_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(base_model))
        ])
        model.fit(X_train, y_train)
        return model, len(df), True
    except Exception as e:
        return None, 0, str(e)

# ==========================================
# 4. Sidebar: Training Data Management
# ==========================================
DEFAULT_TRAIN_FILE = "Training_Data.csv"
CUSTOM_TRAIN_FILE = "Custom_Training_Data.csv"

st.sidebar.title("⚙️ AI Core Settings")
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Training Data Management")

if os.path.exists(CUSTOM_TRAIN_FILE):
    active_file = CUSTOM_TRAIN_FILE
    st.sidebar.success("✅ Status: Using **Custom Training Data**")
    if st.sidebar.button("🗑️ Restore Default EVT Training Data"):
        os.remove(CUSTOM_TRAIN_FILE)
        st.rerun() 
else:
    active_file = DEFAULT_TRAIN_FILE
    if os.path.exists(DEFAULT_TRAIN_FILE):
        st.sidebar.info("ℹ️ Status: Using **Default EVT Training Data**")
    else:
        st.sidebar.error("❌ Cannot find 'Training_Data.csv'. Please ensure it's in your GitHub repo.")
        st.stop()

st.sidebar.markdown("Upload new batch data here to overwrite AI memory and retrain.")
uploaded_train = st.sidebar.file_uploader("Upload New Training Data (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_train is not None:
    try:
        with st.spinner("💾 Saving new training data to server..."):
            if uploaded_train.name.endswith('.csv'):
                new_train_df = pd.read_csv(uploaded_train)
            else:
                new_train_df = pd.read_excel(uploaded_train)
            
            new_train_df.to_csv(CUSTOM_TRAIN_FILE, index=False)
            st.sidebar.success("🎉 Data overwritten successfully! Refreshing AI Brain...")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to save: {e}")

file_mtime = os.path.getmtime(active_file)

with st.spinner('🤖 Loading AI Brain...'):
    model, train_size, status = load_and_train_model(active_file, file_mtime)

if model is None:
    st.error(f"Model training failed! Error: {status}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.metric("🧠 AI Memory Capacity", f"{train_size} units trained")

# ==========================================
# 5. Main Interface: Tabs (Single & Batch)
# ==========================================
st.title("💡 Hip to GTK Calibration AI Predictor (V3.2)")
st.markdown("### 🛡️ Robust Edition: Multi-Sheet Auto Scan & Persistent Training")
st.divider()

tab1, tab2 = st.tabs(["✍️ Single Prediction", "📁 Batch Prediction (File Upload)"])

# --- TAB 1: Single Prediction ---
with tab1:
    st.subheader("Enter Single Hip Test Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1: lv_r = st.number_input("lv-R", value=200.0, step=1.0)
    with col2: lv_g = st.number_input("lv-G", value=390.0, step=1.0)
    with col3: lv_b = st.number_input("lv-B", value=160.0, step=1.0)
    with col4: lv_w = st.number_input("lv-W", value=290.0, step=1.0)
    
    if st.button("🚀 Predict Single GTK Value"):
        input_raw = pd.DataFrame({'lv-R': [lv_r], 'lv-G': [lv_g], 'lv-B': [lv_b], 'lv-W': [lv_w]})
        try:
            X_input, _ = engineer_features(input_raw)
            pred = model.predict(X_input)[0]
            
            st.subheader("🤖 AI Predicted GTK Currents (uA):")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("mix_W_2000_r", f"{pred[0]:.1f}")
            res_col2.metric("mix_W_2000_g", f"{pred[1]:.1f}")
            res_col3.metric("mix_W_2000_b", f"{pred[2]:.1f}")
            res_col4.metric("w_4000_current", f"{pred[3]:.1f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- TAB 2: Batch Prediction ---
def smart_read_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file, low_memory=False)
    
    xls = pd.ExcelFile(uploaded_file)
    possible_r_names = ['lv-R', 'RValue', 'R_Value', 'LightRed_lv', 'lv_R']
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if any(col in df.columns for col in possible_r_names):
            st.toast(f"✅ Auto-locked on sheet: '{sheet_name}'")
            return df
            
        df_offset = pd.read_excel(xls, sheet_name=sheet_name, header=1)
        if any(col in df_offset.columns for col in possible_r_names):
            st.toast(f"✅ Auto-locked on sheet: '{sheet_name}' (Skipped title row)")
            return df_offset
            
    raise ValueError("❌ No valid brightness data (e.g., lv-R) found in any sheet.")

with tab2:
    st.subheader("Upload Data for Batch Prediction")
    uploaded_test = st.file_uploader("Upload Test Data (Foxconn Multi-sheet Excel or CSV)", type=['csv', 'xlsx'])

    if uploaded_test:
        try:
            raw_df = smart_read_file(uploaded_test)
            
            if st.button("🚀 Start Batch Prediction"):
                with st.spinner('AI is cleaning data and predicting...'):
                    
                    X_feat, valid_mask = engineer_features(raw_df)
                    clean_raw_df = raw_df[valid_mask].copy()
                    
                    preds = model.predict(X_feat)
                    
                    sn_candidates = ['SerialNumber', 'SN', 'Hip SN', 'FF SN', 'Serial Number']
                    fatp_candidates = ['FATP Assembly SN', 'AssySn', 'GTK SN']
                    ff_sn_col = next((c for c in sn_candidates if c in clean_raw_df.columns), None)
                    fatp_sn_col = next((c for c in fatp_candidates if c in clean_raw_df.columns), None)
                    
                    output_dict = {
                        'FF SN': clean_raw_df[ff_sn_col].values if ff_sn_col else ['Unknown'] * len(clean_raw_df),
                        'FATP SN': clean_raw_df[fatp_sn_col].values if fatp_sn_col else [''] * len(clean_raw_df),
                        '🎯 predict_w_4000': np.round(preds[:, 3], 1),
                        'predict_mix_W_r': np.round(preds[:, 0], 1),
                        'predict_mix_W_g': np.round(preds[:, 1], 1),
                        'predict_mix_W_b': np.round(preds[:, 2], 1),
                        'lv-R (Raw)': X_feat['lv-R'].values,
                        'lv-G (Raw)': X_feat['lv-G'].values,
                        'lv-B (Raw)': X_feat['lv-B'].values,
                        'lv-W (Raw)': X_feat['lv-W'].values
                    }
                    
                    out_df = pd.DataFrame(output_dict)
                    
                    st.success(f"✅ Prediction complete! Processed {len(out_df)} units. (Dirty data & Upper Limits filtered out)")
                    st.dataframe(out_df.head(15))

                    excel_out = io.BytesIO()
                    out_df.to_excel(excel_out, index=False)
                    st.download_button(
                        label="📥 Download Clean Prediction Results (Excel)",
                        data=excel_out.getvalue(),
                        file_name="AI_Robust_Prediction.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as e:
            st.error(f"❌ Data Parsing Error: {e}")
