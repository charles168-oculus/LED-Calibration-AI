import streamlit as st
import pandas as pd
import io
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="AI Predictor: Hip to GTK V3", layout="wide")
st.title("💡 Hip to GTK Calibration AI Predictor (V3.0)")
st.markdown("### 🚀 Ultra-Robust Version: Auto-Cleaning & Fuzzy Mapping")
st.info("支持 MacBook Numbers 导出文件，自动处理隐藏空格、换行符及大小写差异。")

# ==========================================
# 2. 核心：特征工程函数 (增强鲁棒版)
# ==========================================
def engineer_features(df):
    """自动清洗列名、模糊映射并计算比例"""
    # 预防机制 1: 立即对列名进行深度清洗
    # 去除首尾空格、去除换行符(\n \r)、统一转小写
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r'\n|\r', '', regex=True)
        .str.lower()
    )
    
    # 预防机制 2: 模糊映射表 (全小写匹配)
    mapping = {
        'lv-r': ['lv-r', 'lvr', 'rvalue', 'r_value', 'r'],
        'lv-g': ['lv-g', 'lvg', 'gvalue', 'g_value', 'g'],
        'lv-b': ['lv-b', 'lvb', 'bvalue', 'b_value', 'b'],
        'lv-w': ['lv-w', 'lvw', 'wvalue', 'w_value', 'w']
    }
    
    clean_df = pd.DataFrame()
    for standard_name, candidates in mapping.items():
        # 在清洗后的列名中寻找匹配项
        col = next((c for c in candidates if c in df.columns), None)
        if col:
            # 强制转换为数值，无法转换的变为 NaN
            clean_df[standard_name] = pd.to_numeric(df[col], errors='coerce')
        else:
            available_cols = ", ".join(df.columns.tolist())
            raise ValueError(f"Missing column: '{standard_name}'. \n识别到的表头为: [{available_cols}]")

    # 填充空值为0，防止计算报错
    clean_df = clean_df.fillna(0)

    # 计算比例 (内部逻辑保持小写)
    X = clean_df.copy()
    w_safe = X['lv-w'].replace(0, 1e-5)
    X['r_ratio'] = X['lv-r'] / w_safe
    X['g_ratio'] = X['lv-g'] / w_safe
    X['b_ratio'] = X['lv-b'] / w_safe
    X['rgb_sum'] = X['lv-r'] + X['lv-g'] + X['lv-b']
    
    # 为了输出的一致性，将列名重命名为标准格式
    rename_map = {'lv-r': 'lv-R', 'lv-g': 'lv-G', 'lv-b': 'lv-B', 'lv-w': 'lv-W'}
    X = X.rename(columns=rename_map)
    
    return X

# ==========================================
# 3. 加载并训练模型
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        # 注意：Training_Data 也需要适配新的清洗逻辑
        df_train = pd.read_csv("Training_Data_v2.csv")
        
        # 训练集的目标列名也建议先清洗一下
        df_train.columns = df_train.columns.str.strip().str.lower()
        target_cols = [
            'mix_w_2000_r_ratio_current', 
            'mix_w_2000_g_ratio_current', 
            'mix_w_2000_b_ratio_current', 
            'w_4000_current'
        ]
        
        y = df_train[target_cols]
        X = engineer_features(df_train)
        
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
        feat = engineer_features(input_raw)
        pred = model.predict(feat)[0]
        st.success(f"Results: R:{pred[0]:.4f}, G:{pred[1]:.4f}, B:{pred[2]:.4f}, W4000:{pred[3]:.2f}")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            # 加载数据
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            raw_df = raw_df.reset_index(drop=True)
            
            if st.button("🚀 Run Batch Prediction"):
                # 1. 提取特征并预测 (此处已包含列名自动清洗)
                X_feat = engineer_features(raw_df)
                preds = model.predict(X_feat)
                
                # 2. 构建输出 (SN 匹配也要考虑清洗后的列名)
                # 此时 raw_df.columns 已经被 engineer_features 转换成全小写了
                sn_candidates = ['serialnumber', 'sn', 'hip sn', 'ff sn', '前框sn', 'sf sn']
                sn_col = next((c for c in sn_candidates if c in raw_df.columns), None)
                
                output_dict = {
                    'Hip SN': raw_df[sn_col] if sn_col else ['Unknown_SN']*len(raw_df),
                    'predict_w_4000_current': preds[:, 3],
                    'predict_mix_W_2000_r': preds[:, 0],
                    'predict_mix_W_2000_g': preds[:, 1],
                    'predict_mix_W_2000_b': preds[:, 2],
                    'input_lv-R': X_feat['lv-R'],
                    'input_lv-G': X_feat['lv-G'],
                    'input_lv-B': X_feat['lv-B'],
                    'input_lv-W': X_feat['lv-W']
                }
                out_df = pd.DataFrame(output_dict)
                
                st.success(f"成功处理 {len(out_df)} 行数据!")
                st.dataframe(out_df.head())

                # 3. 导出 Excel
                excel_data = io.BytesIO()
                with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
                    out_df.to_excel(writer, index=False, sheet_name='Predictions')
                
                st.download_button(
                    label="📥 Download Predicted Results",
                    data=excel_data.getvalue(),
                    file_name="AI_Predictions_Result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Error during batch processing: {e}")
