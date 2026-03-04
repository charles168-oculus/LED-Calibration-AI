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
st.set_page_config(page_title="AI Predictor: Hip to GTK V2", layout="wide")
st.title("💡 Hip to GTK Calibration AI Predictor (V2.7)")
st.markdown("Robust Version: Fixed Index Mismatch & Dynamic Column Mapping")

# ==========================================
# 2. 核心：特征工程函数 (带容错)
# ==========================================
def engineer_features(df):
    """自动映射列名并计算比例"""
    # 模糊匹配列名，解决 lv-R 还是 RValue 的问题
    mapping = {
        'lv-R': ['lv-R', 'RValue', 'R_Value', 'R'],
        'lv-G': ['lv-G', 'GValue', 'G_Value', 'G'],
        'lv-B': ['lv-B', 'BValue', 'B_Value', 'B'],
        'lv-W': ['lv-W', 'WValue', 'W_Value', 'W']
    }
    
    clean_df = pd.DataFrame()
    for standard_name, candidates in mapping.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col:
            clean_df[standard_name] = df[col]
        else:
            raise ValueError(f"Missing column: {standard_name}")

    X = clean_df.copy()
    w_safe = X['lv-W'].replace(0, 1e-5)
    X['r_ratio'] = X['lv-R'] / w_safe
    X['g_ratio'] = X['lv-G'] / w_safe
    X['b_ratio'] = X['lv-B'] / w_safe
    X['rgb_sum'] = X['lv-R'] + X['lv-G'] + X['lv-B']
    
    return X

# ==========================================
# 3. 加载并训练模型
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("Training_Data_v2.csv")
        # 目标变量：确保是 4 列
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
        st.error(f"Training Error: {e}")
        return None, False

model, success = load_and_train_model()
if not success: st.stop()

# ==========================================
# 4. 界面切换
# ==========================================
tab1, tab2 = st.tabs(["✍️ Single Prediction", "📁 Batch Prediction"])

with tab1:
    st.subheader("Enter Hip Test Data")
    c1, c2, c3, c4 = st.columns(4)
    lv_r = c1.number_input("lv-R", value=200.0)
    lv_g = c2.number_input("lv-G", value=390.0)
    lv_b = c3.number_input("lv-B", value=160.0)
    lv_w = c4.number_input("lv-W", value=290.0)
    
    if st.button("🚀 Predict Single"):
        input_raw = pd.DataFrame({'lv-R': [lv_r], 'lv-G': [lv_g], 'lv-B': [lv_b], 'lv-W': [lv_w]})
        pred = model.predict(engineer_features(input_raw))[0]
        st.write(f"Results: R:{pred[0]:.1f}, G:{pred[1]:.1f}, B:{pred[2]:.1f}, W4000:{pred[3]:.1f}")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            # 加载并重置索引，防止 Index 报错
            raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            raw_df = raw_df.reset_index(drop=True)
            
            if st.button("🚀 Run Batch Prediction"):
                # 1. 提取特征并预测
                X_feat = engineer_features(raw_df)
                preds = model.predict(X_feat)
                
                # 2. 构建输出 DataFrame (直接构造字典，避免逐列赋值的 Index 冲突)
                sn_candidates = ['SerialNumber', 'SN', 'Hip SN', 'FF SN', '前框SN']
                sn_col = next((c for c in sn_candidates if c in raw_df.columns), None)
                
                output_dict = {
                    'Hip SN': raw_df[sn_col] if sn_col else ['Unknown_SN']*len(raw_df),
                    'GTK SN': [''] * len(raw_df),
                    'predict_w_4000_current': preds[:, 3],
                    'predict_mix_W_2000_r': preds[:, 0],
                    'predict_mix_W_2000_g': preds[:, 1],
                    'predict_mix_W_2000_b': preds[:, 2],
                    'lv-R': X_feat['lv-R'],
                    'lv-G': X_feat['lv-G'],
                    'lv-B': X_feat['lv-B'],
                    'lv-W': X_feat['lv-W']
                }
                out_df = pd.DataFrame(output_dict)
                
                st.success("Success!")
                st.dataframe(out_df.head())

                # 3. Excel 导出逻辑 (略，同之前)
                # ... [此处保留你原有的 Workbook 代码即可] ...
                excel_data = io.BytesIO()
                out_df.to_excel(excel_data, index=False)
                st.download_button("📥 Download Excel", excel_data.getvalue(), "Predict.xlsx")

        except Exception as e:
            st.error(f"Error: {e}")
