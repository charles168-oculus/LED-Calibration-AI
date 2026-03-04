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
st.set_page_config(page_title="AI Predictor: Hip to GTK V3.1", layout="wide")
st.title("💡 Hip to GTK Calibration AI Predictor (V3.1)")
st.markdown("### 🛠 鲁棒修复版：自动处理 MacBook/Numbers 隐藏字符")

# ==========================================
# 2. 核心：特征工程函数 (彻底修复报错逻辑)
# ==========================================
def engineer_features(df):
    """清理列名杂质并计算特征"""
    
    # 【第一步：深度清理表头】
    # 强制去掉所有列名的：首尾空格、换行符(\n)、回车符(\r)
    df.columns = [str(c).strip().replace('\n', '').replace('\r', '') for c in df.columns]
    
    # 【第二步：严格校验】
    required_cols = ['lv-R', 'lv-G', 'lv-B', 'lv-W']
    actual_cols = list(df.columns)
    
    for col in required_cols:
        if col not in df.columns:
            # 如果找不到，直接在界面显示诊断信息
            st.error(f"❌ 缺少必要列: '{col}'")
            st.info(f"📋 系统目前看到的表头是: {actual_cols}")
            st.warning("请确保 Excel 表头文字准确，没有合并单元格。")
            st.stop() # 停止后续运行

    # 【第三步：转换数据】
    clean_df = pd.DataFrame()
    for col in required_cols:
        clean_df[col] = pd.to_numeric(df[col], errors='coerce')

    # 填充缺失值并计算比例
    clean_df = clean_df.fillna(0)
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
        # 加载训练集
        df_train = pd.read_csv("Training_Data_v2.csv")
        
        # 目标列名
        target_cols = [
            'mix_W_2000_r_ratio_current', 
            'mix_W_2000_g_ratio_current', 
            'mix_W_2000_b_ratio_current', 
            'w_4000_current'
        ]
        
        # 清洗训练集并训练
        X = engineer_features(df_train)
        y = df_train[target_cols]
        
        base_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(base_model))
        ])
        model.fit(X, y)
        return model, True
    except Exception as e:
        st.error(f"模型初始化失败: {e}")
        return None, False

model, success = load_and_train_model()
if not success: st.stop()

# ==========================================
# 4. 界面切换
# ==========================================
tab1, tab2 = st.tabs(["✍️ 单一预测", "📁 批量预测"])

with tab1:
    st.subheader("输入测试数据")
    c1, c2, c3, c4 = st.columns(4)
    lv_r = c1.number_input("lv-R", value=200.0)
    lv_g = c2.number_input("lv-G", value=390.0)
    lv_b = c3.number_input("lv-B", value=160.0)
    lv_w = c4.number_input("lv-W", value=290.0)
    
    if st.button("🚀 预测单个"):
        input_raw = pd.DataFrame({'lv-R': [lv_r], 'lv-G': [lv_g], 'lv-B': [lv_b], 'lv-W': [lv_w]})
        feat = engineer_features(input_raw)
        pred = model.predict(feat)[0]
        st.success(f"结果: R:{pred[0]:.4f}, G:{pred[1]:.4f}, B:{pred[2]:.4f}, W4000:{pred[3]:.2f}")

with tab2:
    uploaded_file = st.file_uploader("上传文件 (CSV/XLSX)", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            # 自动读取
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            raw_df = raw_df.reset_index(drop=True)
            
            if st.button("🚀 执行批量预测"):
                # 1. 清理并计算特征
                X_feat = engineer_features(raw_df)
                preds = model.predict(X_feat)
                
                # 2. 匹配 SN (尝试清理后的列名)
                sn_candidates = ['SN', 'SF SN', 'FF SN', 'SerialNumber', '前框SN']
                sn_col = next((c for c in raw_df.columns if c in sn_candidates), None)
                
                output_df = pd.DataFrame({
                    'Hip SN': raw_df[sn_col] if sn_col else ['Unknown']*len(raw_df),
                    'Predict_W4000': preds[:, 3],
                    'Predict_Mix_R': preds[:, 0],
                    'Predict_Mix_G': preds[:, 1],
                    'Predict_Mix_B': preds[:, 2],
                    'Input_lv-R': X_feat['lv-R'],
                    'Input_lv-G': X_feat['lv-G'],
                    'Input_lv-B': X_feat['lv-B'],
                    'Input_lv-W': X_feat['lv-W']
                })
                
                st.success(f"处理完成！共 {len(output_df)} 行。")
                st.dataframe(output_df.head(10))

                # 3. 导出
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='Result')
                
                st.download_button("📥 下载 Excel 结果", output.getvalue(), "AI_Result.xlsx")

        except Exception as e:
            st.error(f"批量处理失败: {e}")
