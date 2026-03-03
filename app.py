import streamlit as st
import pandas as pd
import io
from sklearn.ensemble import RandomForestRegressor
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

# 1. Page Configuration
st.set_page_config(page_title="AI Predictor: Hip to GTK", layout="wide")
st.title("💡 Hip to GTK Calibration AI Predictor")
st.markdown("Choose between manual input for a single prediction, or upload a file for batch prediction.")

# 2. Load and Train Model (Cached)
@st.cache_resource
def load_and_train_model():
    try:
        # Load the training data we prepared earlier
        df = pd.read_csv("Training_Data.csv")
        X = df[['lv-R', 'lv-G', 'lv-B', 'lv-W']]
        y = df[['mix_W_2000_r_ratio_current', 'mix_W_2000_g_ratio_current', 
                'mix_W_2000_b_ratio_current', 'w_4000_current']]
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, True
    except Exception as e:
        return None, False

with st.spinner('Training AI model in background...'):
    model, success = load_and_train_model()

if not success:
    st.error("Could not load 'Training_Data.csv'. Please ensure the file is in the same folder as this script.")
    st.stop()

st.success("✅ Model trained successfully and ready for predictions!")
st.divider()

# 3. Create Tabs for Manual Input and Batch Upload
tab1, tab2 = st.tabs(["✍️ Single Prediction (Manual Input)", "📁 Batch Prediction (File Upload)"])

# ==========================================
# TAB 1: Manual Input
# ==========================================
with tab1:
    st.subheader("Enter Hip Test Data")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lv_r = st.number_input("lv-R", value=200.0, step=1.0)
    with col2:
        lv_g = st.number_input("lv-G", value=390.0, step=1.0)
    with col3:
        lv_b = st.number_input("lv-B", value=160.0, step=1.0)
    with col4:
        lv_w = st.number_input("lv-W", value=290.0, step=1.0)
        
    if st.button("🚀 Predict Single GTK Value"):
        input_data = pd.DataFrame({
            'lv-R': [lv_r],
            'lv-G': [lv_g],
            'lv-B': [lv_b],
            'lv-W': [lv_w]
        })
        
        # Execute prediction
        pred = model.predict(input_data)[0]
        
        st.subheader("🤖 AI Predicted GTK Currents:")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric(label="mix_W_2000_r", value=f"{pred[0]:.2f}")
        res_col2.metric(label="mix_W_2000_g", value=f"{pred[1]:.2f}")
        res_col3.metric(label="mix_W_2000_b", value=f"{pred[2]:.2f}")
        res_col4.metric(label="w_4000_current", value=f"{pred[3]:.2f}")

# ==========================================
# TAB 2: Batch Prediction (File Upload)
# ==========================================
with tab2:
    st.subheader("Upload Data for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
                
            required_cols = ['lv-R', 'lv-G', 'lv-B', 'lv-W']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in the uploaded file: {', '.join(missing_cols)}")
                st.info("Please make sure your file contains columns named exactly: lv-R, lv-G, lv-B, lv-W")
            else:
                st.write("Data loaded successfully! Preview of input data:")
                st.dataframe(input_df.head())
                
                if st.button("🚀 Start Batch Prediction", key="batch_predict"):
                    with st.spinner('AI is predicting GTK values...'):
                        # Extract the required 4 columns
                        X_predict = input_df[required_cols]
                        predictions = model.predict(X_predict)
                        
                        # Construct output DataFrame
                        out_df = pd.DataFrame()
                        
                        # Try to preserve SN if they exist in the uploaded file
                        # 自动寻找可能的 SN 列名
                        possible_sn_cols = ['SerialNumber', 'SN', 'Hip SN', 'FF SN', 'Serial Number']
                        hip_sn_col = next((col for col in possible_sn_cols if col in input_df.columns), None)

                        out_df['Hip SN'] = input_df[hip_sn_col] if hip_sn_col else 'Unknown_SN'
                        out_df['GTK SN'] = input_df['GTK SN'] if 'GTK SN' in input_df.columns else ''
                        
                        # Fill original Hip data
                        out_df['lv-R'] = input_df['lv-R']
                        out_df['lv-G'] = input_df['lv-G']
                        out_df['lv-B'] = input_df['lv-B']
                        out_df['lv-W'] = input_df['lv-W']
                        
                        # Fill predicted GTK data
                        out_df['mix_W_2000_r_ratio_current'] = predictions[:, 0]
                        out_df['mix_W_2000_g_ratio_current'] = predictions[:, 1]
                        out_df['mix_W_2000_b_ratio_current'] = predictions[:, 2]
                        out_df['w_4000_current'] = predictions[:, 3]
                        
                        st.success("✅ Prediction complete!")
                        st.write("Preview of predicted results:")
                        st.dataframe(out_df.head())
                        
                        # Generate formatted Excel file for download
                        wb = Workbook()
                        ws = wb.active
                        ws.title = "Predict_data"
                        
                        # Custom Header row with empty strings for merged cells
                        ws.append(["Hip SN", "GTK SN", "hip test data", "", "", "", "gtk test data", "", "", ""])
                        
                        # Merge cells
                        ws.merge_cells(start_row=1, start_column=3, end_row=1, end_column=6)
                        ws.merge_cells(start_row=1, start_column=7, end_row=1, end_column=10)
                        
                        # Center align the merged headers
                        for col in [1, 2, 3, 7]:
                            ws.cell(row=1, column=col).alignment = Alignment(horizontal='center')
                        
                        # Append actual column names
                        ws.append(list(out_df.columns))
                        
                        # Append data rows
                        for row in dataframe_to_rows(out_df, index=False, header=False):
                            ws.append(row)
                            
                        # Save Excel to memory for download
                        excel_data = io.BytesIO()
                        wb.save(excel_data)
                        excel_data.seek(0)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Predict_data.xlsx",
                            data=excel_data,
                            file_name="Predict_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")
