import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="AI Predictor: Hip to GTK V3.4", layout="wide")
st.title("💡 Hip to GTK Calibration AI Predictor (V3.0)")
st.markdown("### 🛡️ Bulletproof Edition: Auto Multi-Sheet Scan & Dirty Data Cleaning")

# ==========================================
# 2. 核心特征工程与清洗逻辑
# ==========================================
def engineer_features(df):
    """提取核心4列，强制转换为数字，并计算光学比例"""
    mapping = {
        'lv-R': ['lv-R', 'RValue', 'R_Value', 'R-Value', 'LightRed_lv', 'lv-LightRed', 'lv_R'],
        'lv-G': ['lv-G', 'GValue', 'G_Value', 'G-Value', 'LightGreen_lv', 'lv-LightGreen', 'lv_G'],
        'lv-B': ['lv-B', 'BValue', 'B_Value', 'B-Value', 'LightBlue_lv', 'lv-LightBlue', 'lv_B'],
        'lv-W': ['lv-W', 'WValue', 'W_Value', 'W-Value', 'LightWhite_lv', 'lv-LightWhite', 'lv_W']
    }
    
    clean_df = pd.DataFrame(index=df.index)
    found_cols = {}
    
    for standard_name, candidates in mapping.items():
        found_col = next((c for c in candidates if c in df.columns), None)
        if found_col:
            found_cols[standard_name] = found_col
            # 强制转为数字，遇到 "Upper Limit" 这种文本直接变成 NaN
            clean_df[standard_name] = pd.to_numeric(df[found_col], errors='coerce')
        else:
            raise ValueError(f"❌ 找不到必须的亮度列: {standard_name} (请检查表头命名)")

    # 剔除因为文本转换失败而产生的 NaN 行（例如 Upper Limit 行）
    valid_mask = clean_df.notna().all(axis=1)
    clean_df = clean_df[valid_mask].copy()
    
    if len(clean_df) == 0:
        raise ValueError("❌ 清洗后没有找到任何有效的数字测试数据！")

    X = clean_df.copy()
    w_safe = X['lv-W'].replace(0, 1e-5)
    X['r_ratio'] = X['lv-R'] / w_safe
    X['g_ratio'] = X['lv-G'] / w_safe
    X['b_ratio'] = X['lv-B'] / w_safe
    X['rgb_sum'] = X['lv-R'] + X['lv-G'] + X['lv-B']
    
    return X, valid_mask

# ==========================================
# 3. 模型训练 (带缓存)
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        # 注意：这里读取的是你之前上传的 EVT 老数据作为基础大脑
        df = pd.read_csv("Training_Data (1).csv")
        targets = ['mix_W_2000_r_ratio_current', 'mix_W_2000_g_ratio_current', 
                   'mix_W_2000_b_ratio_current', 'w_4000_current']
        df = df.dropna(subset=targets).reset_index(drop=True)
        
        X_train, _ = engineer_features(df)
        y_train = df[targets]
        
        # 使用综合表现最稳的 Gradient Boosting
        base_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(base_model))
        ])
        model.fit(X_train, y_train)
        return model, True
    except Exception as e:
        st.error(f"⚠️ Model Training Failed: {e}")
        return None, False

model, success = load_and_train_model()
if not success: st.stop()

# ==========================================
# 4. 强大的多 Sheet 文件读取引擎
# ==========================================
def smart_read_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file, low_memory=False)
    
    # 如果是 Excel，遍历所有 Sheet 寻找包含亮度数据的表
    xls = pd.ExcelFile(uploaded_file)
    possible_r_names = ['lv-R', 'RValue', 'R_Value', 'LightRed_lv', 'lv_R']
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if any(col in df.columns for col in possible_r_names):
            st.toast(f"✅ 在工作表 '{sheet_name}' 中自动找到了测试数据！")
            return df
            
        # 尝试跳过第一行（合并单元格情况）
        df_offset = pd.read_excel(xls, sheet_name=sheet_name, header=1)
        if any(col in df_offset.columns for col in possible_r_names):
            st.toast(f"✅ 在工作表 '{sheet_name}' (跳过首行) 中找到了测试数据！")
            return df_offset
            
    raise ValueError("❌ 在所有工作表中都找不到前框亮度数据 (如 lv-R, RValue等)。")

# ==========================================
# 5. 批量预测逻辑
# ==========================================
st.subheader("📁 Batch Prediction (Upload Foxconn / Raw Data)")
uploaded_file = st.file_uploader("Upload CSV or Excel (Any format/Multiple sheets)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        raw_df = smart_read_file(uploaded_file)
        
        if st.button("🚀 Run Smart AI Prediction"):
            with st.spinner('AI is cleaning data and predicting...'):
                
                # 1. 特征工程与脏数据过滤
                X_feat, valid_mask = engineer_features(raw_df)
                
                # 过滤出干净的原始数据行
                clean_raw_df = raw_df[valid_mask].copy()
                
                # 2. AI 预测
                preds = model.predict(X_feat)
                
                # 3. 智能提取 SN 信息 (支持 Foxconn 的双 SN 格式)
                sn_candidates = ['SerialNumber', 'SN', 'Hip SN', 'FF SN', 'Serial Number']
                fatp_candidates = ['FATP Assembly SN', 'AssySn', 'GTK SN']
                
                ff_sn_col = next((c for c in sn_candidates if c in clean_raw_df.columns), None)
                fatp_sn_col = next((c for c in fatp_candidates if c in clean_raw_df.columns), None)
                
                # 构建输出
                output_dict = {
                    'FF SN (前框)': clean_raw_df[ff_sn_col].values if ff_sn_col else ['Unknown'] * len(clean_raw_df),
                    'FATP SN (整机)': clean_raw_df[fatp_sn_col].values if fatp_sn_col else [''] * len(clean_raw_df),
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
                
                st.success(f"✅ 成功清洗并预测了 {len(out_df)} 台机器的数据！（已自动过滤 Upper/Lower Limit 等脏数据）")
                st.dataframe(out_df.head(15))

                # 4. 导出为 Excel
                excel_out = io.BytesIO()
                out_df.to_excel(excel_out, index=False)
                st.download_button(
                    label="📥 Download Clean Prediction Results.xlsx",
                    data=excel_out.getvalue(),
                    file_name="AI_Robust_Prediction.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"❌ 解析失败: {e}")
