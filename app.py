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
st.set_page_config(page_title="AI Predictor V3.3", layout="wide")
st.title("💡 Hip to GTK AI Predictor (V3.3)")
st.markdown("### ✅ 结果一致性版：输出列名与原始模板保持一致")

# ==========================================
# 2. 核心函数
# ==========================================
def smart_load_and_clean(file):
    """自动跳过文件名行，精准定位表头"""
    content = file.getvalue().decode('utf-8-sig', errors='ignore').splitlines()
    
    header_idx = 0
    for i, line in enumerate(content[:5]):
        if 'lv-R' in line or 'lv-r' in line:
            header_idx = i
            break
            
    file.seek(0)
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, skiprows=header_idx)
    else:
        df = pd.read_excel(file, skiprows=header_idx)
    
    # 仅在搜索时清洗列名杂质，不改变原始 df 的列名输出需求
    clean_columns = [str(c).strip().replace('\n', '').replace('\r', '') for c in df.columns]
    column_mapping = dict(zip(clean_columns, df.columns))
    return df, column_mapping

def engineer_features(df, cleaned_cols_map):
    """提取特征逻辑，使用映射后的标准名"""
    # 映射表，确保能找到数据
    mapping = {
        'lv-R': next((orig for clean, orig in cleaned_cols_map.items() if clean.lower() == 'lv-r'), None),
        'lv-G': next((orig for clean, orig in cleaned_cols_map.items() if clean.lower() == 'lv-g'), None),
        'lv-B': next((orig for clean, orig in cleaned_cols_map.items() if clean.lower() == 'lv-b'), None),
        'lv-W': next((orig for clean, orig in cleaned_cols_map.items() if clean.lower() == 'lv-w'), None)
    }

    for name, orig_name in mapping.items():
        if not orig_name:
            st.error(f"❌ 找不到必要列: {name}")
            st.stop()

    X = pd.DataFrame()
    X['lv-R'] = pd.to_numeric(df[mapping['lv-R']], errors='coerce').fillna(0)
    X['lv-G'] = pd.to_numeric(df[mapping['lv-G']], errors='coerce').fillna(0)
    X['lv-B'] = pd.to_numeric(df[mapping['lv-B']], errors='coerce').fillna(0)
    X['lv-W'] = pd.to_numeric(df[mapping['lv-W']], errors='coerce').fillna(0)

    w_safe = X['lv-W'].replace(0, 1e-5)
    X['r_ratio'] = X['lv-R'] / w_safe
    X['g_ratio'] = X['lv-G'] / w_safe
    X['b_ratio'] = X['lv-B'] / w_safe
    X['rgb_sum'] = X['lv-R'] + X['lv-G'] + X['lv-B']
    return X

# ==========================================
# 3. 模型训练
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        df_train = pd.read_csv("Training_Data_v2.csv")
        # 训练集清洗
        train_clean_map = {str(c).strip().replace('\n', ''): c for c in df_train.columns}
        
        # 目标列名（请确保与训练集 csv 完全一致）
        target_cols = [
            'mix_W_2000_r_ratio_current', 
            'mix_W_2000_g_ratio_current', 
            'mix_W_2000_b_ratio_current', 
            'w_4000_current'
        ]
        
        X = engineer_features(df_train, train_clean_map)
        y = df_train[target_cols]
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, random_state=42)))
        ])
        model.fit(X, y)
        return model, True
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, False

model, success = load_and_train_model()
if not success: st.stop()

# ==========================================
# 4. 界面
# ==========================================
tab1, tab2 = st.tabs(["✍️ Single Prediction", "📁 Batch Prediction"])

with tab1:
    # 单一预测保持原样
    st.subheader("Manual Input")
    c = st.columns(4)
    vals = [c[i].number_input(n, value=v) for i, (n, v) in enumerate(zip(['lv-R','lv-G','lv-B','lv-W'], [200.0, 390.0, 160.0, 290.0]))]
    if st.button("🚀 Predict"):
        input_df = pd.DataFrame([vals], columns=['lv-R','lv-G','lv-B','lv-W'])
        res = model.predict(engineer_features(input_df, {n:n for n in input_df.columns}))[0]
        st.write(f"Results: R:{res[0]:.4f}, G:{res[1]:.4f}, B:{res[2]:.4f}, W400:{res[3]:.2f}")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            # 智能读取
            raw_df, clean_map = smart_load_and_clean(uploaded_file)
            
            if st.button("🚀 Run Batch Prediction"):
                # 1. 预测
                X_feat = engineer_features(raw_df, clean_map)
                preds = model.predict(X_feat)
                
                # 2. 构造输出：直接修改原 df 的目标列，保持列名一致
                # 定义你的标准目标列名
                target_names = [
                    'mix_W_2000_r_ratio_current', 
                    'mix_W_2000_g_ratio_current', 
                    'mix_W_2000_b_ratio_current', 
                    'w_4000_current'
                ]
                
                # 将预测值填入
                raw_df[target_names[0]] = preds[:, 0]
                raw_df[target_names[1]] = preds[:, 1]
                raw_df[target_names[2]] = preds[:, 2]
                raw_df[target_names[3]] = preds[:, 3]
                
                st.success("预测完成！")
                st.dataframe(raw_df.head())
                
                # 3. 导出包含所有原始列的文件
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    raw_df.to_excel(writer, index=False)
                
                st.download_button("📥 Download Result Excel", output.getvalue(), "AI_Batch_Result.xlsx")
        except Exception as e:
            st.error(f"处理出错: {e}")
