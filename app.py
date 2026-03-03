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
st.title("💡 Hip to GTK Calibration AI Predictor (V2.0)")
st.markdown("Enhanced with Ratio-Features and Gradient Boosting to break prediction ceilings.")

# ==========================================
# 2. 核心：特征工程函数 (引入发光比例)
# ==========================================
def engineer_features(df):
    """自动计算发光比例，这比单纯看亮度绝对值准得多"""
    X = df[['lv-R', 'lv-G', 'lv-B', 'lv-W']].copy()
    
    # 避免分母为0
    w_safe = X['lv-W'].replace(0, 1e-5)
    
    X['r_ratio'] = X['lv-R'] / w_safe
    X['g_ratio'] = X['lv-G'] / w_safe
    X['b_ratio'] = X['lv-B'] / w_safe
    
    # 还可以加入总亮度特征
    X['rgb_sum'] = X['lv-R'] + X['lv-G'] + X['lv-B']
    
    return X

# ==========================================
# 3. 加载并训练模型 (带缓存)
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        # 注意：这里改成了读取我们刚刚新生成的 v2 版数据
        df = pd.read_csv("Training_Data_v2.csv")
        
        # 提取目标变量
        y = df[['mix_W_2000_r_ratio_current', 'mix_W_2000_g_ratio_current', 
                'mix_W_2000_b_ratio_current', 'w_4000_current']]
        
        # 提取并构建增强特征
        X = engineer_features(df)
        
        # 使用可以突破最大值天花板的 Gradient Boosting 算法
        base_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(base_model))
        ])
        
        model.fit(X, y)
        return model, True
    except Exception as e:
        return None, False

with st.spinner('Training enhanced AI model with DOE1 data...'):
    model, success = load_and_train_model()

if not success:
    st.error("Could not load 'Training_Data_v2.csv'. Please make sure you downloaded the new file and put it in the same folder.")
    st.stop()

st.success("✅ Enhanced AI Model trained successfully! (Ready for extreme values)")
st.divider()

# ==========================================
# 4. 界面切换: 单数据输入 vs 批量上传
# ==========================================
tab1, tab2 = st.tabs(["✍️ Single Prediction", "📁 Batch Prediction (File Upload)"])

with tab1:
    st.subheader("Enter Hip Test Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1: lv_r = st.number_input("lv-R", value=200.0, step=1.0)
    with col2: lv_g = st.number_input("lv-G", value=390.0, step=1.0)
    with col3: lv_b = st.number_input("lv-B", value=160.0, step=1.0)
    with col4: lv_w = st.number_input("lv-W", value=290.0, step=1.0)
        
    if st.button("🚀 Predict Single GTK Value"):
        input_raw = pd.DataFrame({'lv-R': [lv_r], 'lv-G': [lv_g], 'lv-B': [lv_b], 'lv-W': [lv_w]})
        X_input = engineer_features(input_raw)
        pred = model.predict(X_input)[0]
        
        st.subheader("🤖 AI Predicted GTK Currents:")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("mix_W_2000_r", f"{pred[0]:.2f}")
        res_col2.metric("mix_W_2000_g", f"{pred[1]:.2f}")
        res_col3.metric("mix_W_2000_b", f"{pred[2]:.2f}")
        res_col4.metric("w_4000_current", f"{pred[3]:.2f}")

with tab2:
    st.subheader("Upload Data for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
                
            required_cols = ['lv-R', 'lv-G', 'lv-B', 'lv-W']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                st.write("Data loaded successfully! Preview:")
                st.dataframe(input_df.head())
                
                if st.button("🚀 Start Batch Prediction", key="batch_predict"):
                    with st.spinner('AI is predicting...'):
                        # 转换并预测
                        X_predict = engineer_features(input_df)
                        predictions = model.predict(X_predict)
                        
                        out_df = pd.DataFrame()
                        
                        # 【修复 Bug 1】：超级聪明的 SN 提取逻辑，涵盖几乎所有叫法
                        possible_hip_sn = ['SerialNumber', 'SN', 'Hip SN', 'FF SN', 'Serial Number', '前框SN']
                        hip_col = next((c for c in possible_hip_sn if c in input_df.columns), None)
                        possible_gtk_sn = ['GTK SN', 'SF SN', '整机SN']
                        gtk_col = next((c for c in possible_gtk_sn if c in input_df.columns), None)
                        
                        out_df['Hip SN'] = input_df[hip_col] if hip_col else 'Unknown_SN'
                        out_df['GTK SN'] = input_df[gtk_col] if gtk_col else ''
                        
                        out_df['lv-R'] = input_df['lv-R']
                        out_df['lv-G'] = input_df['lv-G']
                        out_df['lv-B'] = input_df['lv-B']
                        out_df['lv-W'] = input_df['lv-W']
                        
                        out_df['mix_W_2000_r_ratio_current'] = predictions[:, 0]
                        out_df['mix_W_2000_g_ratio_current'] = predictions[:, 1]
                        out_df['mix_W_2000_b_ratio_current'] = predictions[:, 2]
                        out_df['w_4000_current'] = predictions[:, 3]
                        
                        st.success("✅ Prediction complete!")
                        st.dataframe(out_df.head())
                        
                        # 生成可下载的 Excel
                        wb = Workbook()
                        ws = wb.active
                        ws.title = "Predict_data"
                        
                        ws.append(["Hip SN", "GTK SN", "hip test data", "", "", "", "gtk test data", "", "", ""])
                        ws.merge_cells(start_row=1, start_column=3, end_row=1, end_column=6)
                        ws.merge_cells(start_row=1, start_column=7, end_row=1, end_column=10)
                        
                        for col in [1, 2, 3, 7]:
                            ws.cell(row=1, column=col).alignment = Alignment(horizontal='center')
                        
                        ws.append(list(out_df.columns))
                        for row in dataframe_to_rows(out_df, index=False, header=False):
                            ws.append(row)
                            
                        excel_data = io.BytesIO()
                        wb.save(excel_data)
                        excel_data.seek(0)
                        
                        st.download_button(
                            label="📥 Download Predict_data.xlsx",
                            data=excel_data,
                            file_name="Predict_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        except Exception as e:
            st.error(f"Error processing file: {e}")
