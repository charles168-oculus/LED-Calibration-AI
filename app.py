import streamlit as st
import pandas as pd
import io
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="AI Predictor: Hip to GTK V2.7", layout="wide")
st.title("💡 Hip to GTK Calibration AI Predictor (V2.7)")
st.markdown("### Auto-detects Header Rows | Robust Data Alignment | Batch Prediction")

# ==========================================
# 2. 核心特征工程函数
# ==========================================
def engineer_features(df):
    """提取核心4列并计算光学比例"""
    # 模糊匹配映射表
    mapping = {
        'lv-R': ['lv-R', 'RValue', 'R_Value', 'R-Value'],
        'lv-G': ['lv-G', 'GValue', 'G_Value', 'G-Value'],
        'lv-B': ['lv-B', 'BValue', 'B_Value', 'B-Value'],
        'lv-W': ['lv-W', 'WValue', 'W_Value', 'W-Value']
    }
    
    clean_df = pd.DataFrame()
    for standard_name, candidates in mapping.items():
        # 在当前 df 列名中寻找匹配项
        found_col = next((c for c in candidates if c in df.columns), None)
        if found_col:
            clean_df[standard_name] = pd.to_numeric(df[found_col], errors='coerce').fillna(0)
        else:
            raise ValueError(f"Missing required column: {standard_name}")

    X = clean_df.copy()
    w_safe = X['lv-W'].replace(0, 1e-5)
    X['r_ratio'] = X['lv-R'] / w_safe
    X['g_ratio'] = X['lv-G'] / w_safe
    X['b_ratio'] = X['lv-B'] / w_safe
    X['rgb_sum'] = X['lv-R'] + X['lv-G'] + X['lv-B']
    
    return X

# ==========================================
# 3. 模型训练 (带缓存)
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        # 确保 Training_Data_v2.csv 存在于同级目录
        df = pd.read_csv("Training_Data_v2.csv")
        y = df[['mix_W_2000_r_ratio_current', 'mix_W_2000_g_ratio_current', 
                'mix_W_2000_b_ratio_current', 'w_4000_current']]
        X = engineer_features(df)
        
        base_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(base_model))
        ])
        model.fit(X, y)
        return model, True
    except Exception as e:
        st.error(f"Model Training Failed: {e}")
        return None, False

model, success = load_and_train_model()
if not success: st.stop()

# ==========================================
# 4. 批量预测逻辑
# ==========================================
st.subheader("📁 Batch Prediction (File Upload)")
uploaded_file = st.file_uploader("Upload Hip Test Report (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # --- 关键：智能表头识别逻辑 ---
        # 先试读第一行看看有没有 lv-R
        preview = pd.read_excel(uploaded_file, nrows=1) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file, nrows=1)
        
        header_row = 0
        # 如果第一行没找到 lv-R 或 RValue，说明第一行可能是合并单元格标题，跳到第二行读取
        if not any(col in preview.columns for col in ['lv-R', 'RValue', 'R_Value']):
            header_row = 1
            st.info("ℹ️ First row detected as Title/Merged cell. Shifting to second row for headers.")

        # 正式读取
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file, header=header_row)
        else:
            input_df = pd.read_excel(uploaded_file, header=header_row)

        # 清洗：去掉全是空的行，并重置索引防止崩溃
        input_df = input_df.dropna(how='all').reset_index(drop=True)
        
        st.write(f"Loaded {len(input_df)} rows. Columns found:", list(input_df.columns))

        if st.button("🚀 Run AI Prediction"):
            with st.spinner('AI is calculating...'):
                # 1. 特征工程
                X_feat = engineer_features(input_df)
                
                # 2. AI 预测
                preds = model.predict(X_feat)
                
                # 3. 构建输出字典 (使用 dict 构造法最稳定)
                sn_candidates = ['SerialNumber', 'SN', 'Hip SN', 'FF SN', 'Serial Number', '前框SN']
                sn_col = next((c for c in sn_candidates if c in input_df.columns), None)
                
                # 强制对齐长度
                output_dict = {
                    'Hip SN': input_df[sn_col].values if sn_col else ['Unknown_SN'] * len(input_df),
