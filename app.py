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
st.title("💡 Hip to GTK Calibration AI Predictor (V2.6)")
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
            # 1. 自动识别文件格式并加载
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
            
            # 2. 强力清洗数据：去除全空行，防止索引干扰
            input_df = input_df.dropna(how='all').reset_index(drop=True)

            # 3. 模糊匹配核心列 (解决 lv-R 还是 RValue 的问题)
            # 定义搜索映射：将你模型需要的标准名 映射到 可能出现的各种列名
            mapping = {
                'lv-R': ['lv-R', 'RValue', 'R_Value', 'lv_r'],
                'lv-G': ['lv-G', 'GValue', 'G_Value', 'lv_g'],
                'lv-B': ['lv-B', 'BValue', 'B_Value', 'lv_b'],
                'lv-W': ['lv-W', 'WValue', 'W_Value', 'lv_w']
            }
            
            # 自动寻找存在的列
            found_cols = {}
            for target, candidates in mapping.items():
                for c in candidates:
                    if c in input_df.columns:
                        found_cols[target] = c
                        break
            
            # 检查是否找全了 4 个亮度指标
            if len(found_cols) < 4:
                st.error(f"Missing brightness data. Need: lv-R, lv-G, lv-B, lv-W. Found: {list(found_cols.keys())}")
                st.stop()
            
            # 4. 预测与导出逻辑
            if st.button("🚀 Start Batch Prediction", key="batch_predict"):
                with st.spinner('AI is predicting...'):
                    # 准备预测专用的输入 (严格按照模型需要的 4 列提取)
                    predict_input = pd.DataFrame({
                        'lv-R': input_df[found_cols['lv-R']],
                        'lv-G': input_df[found_cols['lv-G']],
                        'lv-B': input_df[found_cols['lv-B']],
                        'lv-W': input_df[found_cols['lv-W']]
                    })
                    
                    # 运行特征工程和预测
                    X_predict = engineer_features(predict_input)
                    predictions = model.predict(X_predict)
                    
                    # 【核心修复】：构建输出表，强制对齐索引
                    out_df = pd.DataFrame(index=input_df.index) 

                    # 提取 SN (容错逻辑)
                    sn_candidates = ['SerialNumber', 'SN', 'Hip SN', 'FF SN', 'Serial Number', '前框SN']
                    hip_col = next((c for c in sn_candidates if c in input_df.columns), None)
                    out_df['Hip SN'] = input_df[hip_col] if hip_col else 'Unknown_SN'
                    out_df['GTK SN'] = '' # 预测阶段 GTK 为空
                    
                    # 将预测结果填入
                    out_df['predict_w_4000_current'] = predictions[:, 3] # 对应你 y 的第 4 列
                    out_df['predict_mix_W_2000_r'] = predictions[:, 0]
                    out_df['predict_mix_W_2000_g'] = predictions[:, 1]
                    out_df['predict_mix_W_2000_b'] = predictions[:, 2]
                    
                    # 补充原始数据方便核对
                    out_df['lv-R'] = predict_input['lv-R']
                    out_df['lv-G'] = predict_input['lv-G']
                    out_df['lv-B'] = predict_input['lv-B']
                    out_df['lv-W'] = predict_input['lv-W']
                    
                    st.success("✅ Batch prediction finished!")
                    st.dataframe(out_df.head())
                    
                    # 5. 生成 Excel (带样式的导出逻辑)
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "AI_Prediction_Result"
                    # 添加双行表头
                    ws.append(["SN Info", "", "Predicted GTK Target Current (mA)", "", "", "", "Original Hip Data", "", "", ""])
                    ws.append(list(out_df.columns))
                    for r in dataframe_to_rows(out_df, index=False, header=False):
                        ws.append(r)
                    
                    # 导出
                    excel_data = io.BytesIO()
                    wb.save(excel_data)
                    st.download_button("📥 Download Excel Report", excel_data.getvalue(), 
                                       "Predict_Result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Runtime Error: {str(e)}")
