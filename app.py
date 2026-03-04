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
st.set_page_config(page_title="AI Predictor V3.2", layout="wide")
st.title("💡 Hip to GTK AI Predictor (V3.2)")
st.markdown("### 🛠️ 智能表头定位版：解决多余行与隐藏字符")

# ==========================================
# 2. 核心函数：智能读取并提取特征
# ==========================================
def smart_load_and_clean(file):
    """智能跳过干扰行，并清洗表头"""
    # 先以文本形式读几行看看
    content = file.getvalue().decode('utf-8-sig', errors='ignore').splitlines()
    
    # 寻找包含 "lv-R" 的那一行作为真正的起始行
    header_idx = 0
    for i, line in enumerate(content[:5]): # 只检查前5行
        if 'lv-R' in line:
            header_idx = i
            break
            
    # 重新读取数据
    file.seek(0)
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, skiprows=header_idx)
    else:
        df = pd.read_excel(file, skiprows=header_idx)
    
    # 清理表头隐藏字符
    df.columns = [str(c).strip().replace('\n', '').replace('\r', '') for c in df.columns]
    return df

def engineer_features(df):
    """计算特征逻辑"""
    required_cols = ['lv-R', 'lv-G', 'lv-B', 'lv-W']
    
    # 容错：如果没找到，尝试从小写里找
    for col in required_cols:
        if col not in df.columns:
            # 尝试不区分大小写
            mapping = {c.lower(): c for c in df.columns}
            if col.lower() in mapping:
                df = df.rename(columns={mapping[col.lower()]: col})
            else:
                st.error(f"❌ 找不到列: '{col}'。")
                st.info(f"📋 当前表头: {list(df.columns)}")
                st.stop()

    clean_df = pd.DataFrame()
    for col in required_cols:
        clean_df[col] = pd.to_numeric(df[col], errors='coerce')

    clean_df = clean_df.fillna(0)
    X = clean_df.copy()
    w_safe = X['lv-W'].replace(0, 1e-5)
    X['r_ratio'] = X['lv-R'] / w_safe
    X['g_ratio'] = X['lv-G'] / w_safe
    X['b_ratio'] = X['lv-B'] / w_safe
    X['rgb_sum'] = X['lv-R'] + X['lv-G'] + X['lv-B']
    return X

# ==========================================
# 3. 模型加载 (逻辑同前，调用新清洗函数)
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        # 这里假设训练集是标准的，如果不是，也得跳行
        df_train = pd.read_csv("Training_Data_v2.csv")
        # 兼容性清洗
        df_train.columns = [str(c).strip().replace('\n', '').replace('\r', '') for c in df_train.columns]
        
        target_cols = ['mix_W_2000_r_ratio_current', 'mix_W_2000_g_ratio_current', 
                       'mix_W_2000_b_ratio_current', 'w_4000_current']
        
        X = engineer_features(df_train)
        y = df_train[target_cols]
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, random_state=42)))
        ])
        model.fit(X, y)
        return model, True
    except Exception as e:
        st.error(f"模型初始化失败: {e}")
        return None, False

model, success = load_and_train_model()
if not success: st.stop()

# ==========================================
# 4. 界面
# ==========================================
tab1, tab2 = st.tabs(["✍️ 单个预测", "📁 批量预测"])

with tab1:
    # 单个预测逻辑保持不变...
    st.subheader("输入数值")
    c1,c2,c3,c4 = st.columns(4)
    lv_r = c1.number_input("lv-R", value=200.0)
    lv_g = c2.number_input("lv-G", value=390.0)
    lv_b = c3.number_input("lv-B", value=160.0)
    lv_w = c4.number_input("lv-W", value=290.0)
    if st.button("🚀 预测"):
        input_df = pd.DataFrame({'lv-R': [lv_r], 'lv-G': [lv_g], 'lv-B': [lv_b], 'lv-W': [lv_w]})
        res = model.predict(engineer_features(input_df))[0]
        st.success(f"结果: R:{res[0]:.4f}, G:{res[1]:.4f}, B:{res[2]:.4f}, W4000:{res[3]:.2f}")

with tab2:
    uploaded_file = st.file_uploader("上传 CSV/Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            # 【关键改变】使用智能读取函数
            raw_df = smart_load_and_clean(uploaded_file)
            
            if st.button("🚀 开始批量处理"):
                X_feat = engineer_features(raw_df)
                preds = model.predict(X_feat)
                
                # 寻找 SN
                sn_candidates = ['SN', 'SF SN', 'FF SN', 'SerialNumber']
                sn_col = next((c for c in raw_df.columns if c in sn_candidates), None)
                
                out_df = pd.DataFrame({
                    'SN': raw_df[sn_col] if sn_col else "Unknown",
                    'Predict_W4000': preds[:, 3],
                    'Predict_Mix_R': preds[:, 0],
                    'Predict_Mix_G': preds[:, 1],
                    'Predict_Mix_B': preds[:, 2]
                })
                
                st.dataframe(out_df.head())
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    out_df.to_excel(writer, index=False)
                st.download_button("📥 下载结果", output.getvalue(), "Result.xlsx")
        except Exception as e:
            st.error(f"处理失败: {e}")
