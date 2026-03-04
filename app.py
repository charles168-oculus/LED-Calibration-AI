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
st.markdown("### 🛠 透明逻辑版：自动清理隐藏换行与空格")

# ==========================================
# 2. 核心：特征工程函数 (透明清理逻辑)
# ==========================================
def engineer_features(df):
    """只清理看不见的杂质，不修改列名内容"""
    
    # 【核心修复】强制清理表头所有的首尾空格、换行符
    df.columns = [str(c).strip().replace('\n', '').replace('\r', '') for c in df.columns]
    
    # 定义你程序要求的标准列名
    required_cols = ['lv-R', 'lv-G', 'lv-B', 'lv-W']
    
    # 检查列是否存在，不存在则给出极其详细的报错
    for col in required_cols:
        if col not in df.columns:
            actual_cols = list(df.columns)
            st.error(f"❌ 找不到列: '{col}'")
            st.info(f"📋 当前文件识别到的表头为: {actual_cols}")
            st.warning("提示：请检查 Excel 中该列表头是否拼写准确，或是否存在合并单元格。")
            st.stop()

    # 提取数据并确保是数字格式
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
        
        # 训练集的目标列 (请确保 Training_Data_v2.csv 里这四列名字是对的)
        target_cols = [
            'mix_W_2000_r_ratio_current', 
            'mix_W_2000_g_ratio_current', 
            'mix_W_2000_b_ratio_current', 
            'w_4000_current'
        ]
        
        # 使用同样的特征提取函数进行清洗
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
tab1, tab2 = st.tabs(["✍️ 单一预测 (Manual)", "📁 批量预测 (Batch)"])

with tab1:
    st.subheader("输入 Hip 测试数据")
    c1, c2, c3, c4 = st.columns(4)
    lv_r = c1.number_input("lv-R", value=200.0)
    lv_g = c2.number_input("lv-G", value=390.0)
    lv_b = c3.number_input("lv-B", value=160.0)
    lv_w = c4.number_input("lv-W", value=290.0)
    
    if st.button("🚀 开始预测"):
        input_raw = pd.DataFrame({'lv-R': [lv_r], 'lv-G': [lv_g], 'lv-B': [lv_b], 'lv-W': [lv_w]})
        feat = engineer_features(input_raw)
        pred = model.predict(feat)[0]
        st.success(f"结果: R:{pred[0]:.4f}, G:{pred[1]:.4f}, B:{pred[2]:.4f}, W4000:{pred[3]:.2f}")

with tab2:
    uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            # 自动识别格式读取
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            raw_df = raw_df.reset_index(drop=True)
            
            if st.button("🚀 执行批量预测"):
                # 1. 清洗并预测
                X_feat = engineer_features(raw_df)
                preds = model.predict(X_feat)
                
                # 2. 匹配 SN (尝试几种常见的 SN 命名)
                sn_candidates = ['SN', 'SF SN', 'FF SN', 'SerialNumber', '前框SN']
                # 同样对 sn_candidates 进行清洗匹配
                sn_col = next((c for c in raw_df.columns if c in sn_candidates), None)
                
                output_df = pd.DataFrame({
                    'Hip SN': raw_df[sn_col] if sn_col else ['Unknown']*len(raw_df),
                    'Predict_W4000': preds[:, 3],
                    'Predict_Mix_R': preds[:, 0],
                    'Predict_Mix_G': preds[:, 1],
                    'Predict_Mix_B': preds[:, 2],
                    'Original_lv-R': X_feat['lv-R'],
                    'Original_lv-G': X_feat['lv-G'],
                    'Original_lv-B': X_feat['lv-B'],
                    'Original_lv-W': X_feat['lv-W']
                })
                
                st.success(f"处理完成！共 {len(output_df)} 行。")
                st.dataframe(output_df.head(10))

                # 3. 导出
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='Result')
                
                st.download_button(
                    label="📥 下载结果 Excel",
                    data=output.getvalue(),
                    file_name="AI_Calibration_Result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"批量处理出错: {e}")
