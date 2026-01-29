import streamlit as st
import pandas as pd
import io
import os
from reconciliation_core import clean_currency, normalize_col, find_best_match, reconcile_data

st.set_page_config(page_title="Reconciliation System", layout="wide")

# --- THEME TOGGLE (Night/Day Mode) ---
with st.sidebar:
    st.write("### 🎨 Appearance")
    # Toggle switch: Default False (Day Mode), True (Night Mode)
    is_night_mode = st.toggle("🌙 Night Mode", value=False)

if is_night_mode:
    # Force Dark Mode CSS
    st.markdown("""
    <style>
        /* Main App Background and Text */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #262730;
        }
        /* Headers */
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, p, label {
            color: #FAFAFA !important;
        }
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #FAFAFA !important;
        }
        /* DataFrames (attempt to dark mode) */
        .stDataFrame {
            filter: invert(0.9) hue-rotate(180deg);
        }
        /* Input Fields */
        .stTextInput > div > div > input {
            color: #FAFAFA;
            background-color: #262730;
        }
        .stSelectbox > div > div > div {
            color: #FAFAFA;
            background-color: #262730;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Force Light Mode CSS (Day Mode - "Clean Slate" Professional Theme)
    st.markdown("""
    <style>
        /* Force Streamlit Theme Variables for Light Mode */
        :root {
            --primary-color: #091E42;
            --background-color: #FFFFFF;
            --secondary-background-color: #F7F9FC;
            --text-color: #172B4D;
            --font: sans-serif;
        }

        /* Main App Background - Clean White */
        .stApp {
            background-color: #FFFFFF;
            color: #172B4D; /* Deep Blue-Grey (Enterprise Standard) */
        }
        
        /* Force Text Colors for All Elements */
        p, div, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
            color: #172B4D !important;
        }

        /* Sidebar Background - Cool Light Gray */
        [data-testid="stSidebar"] {
            background-color: #F7F9FC;
            border-right: 1px solid #DFE1E6;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
             color: #172B4D !important;
        }

        /* Input Fields - White Background, Dark Text, Subtle Border */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input {
            color: #172B4D !important;
            background-color: #FFFFFF !important;
            border: 1px solid #DFE1E6;
        }
        
        /* Fix Dropdown Menu Options Visibility */
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #FFFFFF !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            color: #172B4D !important;
            background-color: #FFFFFF !important;
        }
        /* Hover state for options */
        ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
            background-color: #F0F2F6 !important;
        }
        
        /* Alert Boxes - Ensure Text Contrast */
        .stAlert div {
            color: #172B4D !important;
        }

        /* File Uploader Fix - Ensure Dropzone Text is Readable */
        [data-testid="stFileUploader"] section {
            background-color: #F7F9FC; /* Light gray dropzone */
            border: 1px dashed #DFE1E6;
        }
        /* Target the main text "Drag and drop file here" */
        [data-testid="stFileUploader"] section span, 
        [data-testid="stFileUploader"] section div {
            color: #172B4D !important;
        }
        /* Target the small text "Limit 200MB per file..." */
        [data-testid="stFileUploader"] small {
            color: #5E6C84 !important;
        }
        /* Ensure button text is readable */
        [data-testid="stFileUploader"] button {
            color: #172B4D !important;
            border-color: #DFE1E6 !important;
            background-color: #FFFFFF !important; /* Force White Background */
        }
        
        /* Universal Button Fix (Download, Submit, etc.) */
        button {
            color: #172B4D !important;
        }
        .stButton > button, .stDownloadButton > button {
            color: #172B4D !important;
            background-color: #FFFFFF !important;
            border: 1px solid #DFE1E6 !important;
        }
        
        /* Hover Effects for Buttons */
        [data-testid="stFileUploader"] button:hover,
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #172B4D !important;
            background-color: #F0F2F6 !important;
            color: #172B4D !important;
        }

        /* Fix DataFrame Toolbar and Header */
        [data-testid="stDataFrame"] {
            background-color: #262730 !important;
        }
        [data-testid="stDataFrame"] div {
            color: #FFFFFF !important;
        }
        
        /* Streamlit Toolbar (Top Right) */
        [data-testid="stToolbar"] {
            color: #172B4D !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Financial Reconciliation System")
st.markdown("Upload your Accounting and Budget Excel files to reconcile them based on **ORS No.** and **Common Columns**.")

# Helper to get local files
def get_excel_files():
    try:
        cwd = os.getcwd()
        files = [f for f in os.listdir(cwd) if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
        return files
    except Exception:
        return []

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Data Source")

st.sidebar.info("💡 **Tip:** Ensure your Excel files are **CLOSED** before uploading to avoid lock errors.")

# Upload Section
st.sidebar.subheader("📤 Upload Files")
st.sidebar.caption("Supported formats: .xlsx, .xls, .csv, .xlsm, .xlsb")

uploaded_acc = st.sidebar.file_uploader(
    "Accounting File", 
    type=["xlsx", "xls", "csv", "xlsm", "xlsb"], 
    key="acc_file_v3"
)
uploaded_bud = st.sidebar.file_uploader(
    "Budget File", 
    type=["xlsx", "xls", "csv", "xlsm", "xlsb"], 
    key="bud_file_v3"
)

# Local Files Section
st.sidebar.divider()
st.sidebar.subheader("📂 OR Use Local Files")
local_files = get_excel_files()
use_local = st.sidebar.checkbox("Enable Local File Selection", value=False)

acc_local = None
bud_local = None

if use_local:
    if not local_files:
        st.sidebar.info("No Excel files found in directory.")
    else:
        # Default indices
        idx_acc = 0
        idx_bud = 1 if len(local_files) > 1 else 0
        acc_local = st.sidebar.selectbox("Select Accounting (Local)", local_files, index=idx_acc, key="acc_local")
        bud_local = st.sidebar.selectbox("Select Budget (Local)", local_files, index=idx_bud, key="bud_local")

accounting_file = None
budget_file = None

# Logic: Priority to Uploaded, then Local
if uploaded_acc is not None:
    accounting_file = uploaded_acc
elif use_local and acc_local:
    accounting_file = acc_local

if uploaded_bud is not None:
    budget_file = uploaded_bud
elif use_local and bud_local:
    budget_file = bud_local

def load_data(file):
    """Loads Excel data using strict engine selection for stability."""
    # Ensure pointer is at start if it's a file-like object
    if hasattr(file, 'seek'):
        file.seek(0)
    
    filename = file.name.lower() if hasattr(file, 'name') else ""

    # 1. Handle CSV
    if filename.endswith('.csv'):
        return pd.read_csv(file)

    # 2. Strict .xls handling (Use xlrd ONLY)
    if filename.endswith('.xls'):
        try:
            return pd.read_excel(file, engine="xlrd")
        except Exception:
            if hasattr(file, 'seek'): file.seek(0)
            return pd.read_excel(file, engine="calamine")

    # 3. Strict .xlsx handling (Use openpyxl ONLY)
    if filename.endswith(('.xlsx', '.xlsm', '.xltx', '.xltm')):
        try:
            return pd.read_excel(file, engine="openpyxl")
        except Exception:
            if hasattr(file, 'seek'): file.seek(0)
            return pd.read_excel(file, engine="calamine")

    # 4. Fallback for unknown extensions
    return pd.read_excel(file, engine="calamine") 

if accounting_file and budget_file:
    st.divider()
    st.info("🚀 Starting File Processing...")
    
    try:
        # Load Data
        df_acc = load_data(accounting_file)
        st.write(f"✅ Loaded Accounting File: {len(df_acc):,} rows")
        
        df_bud = load_data(budget_file)
        st.write(f"✅ Loaded Budget File: {len(df_bud):,} rows")

        # Display source info
        if uploaded_acc or uploaded_bud:
            st.success("Files uploaded successfully!")
        else:
            st.success(f"Using local files: {accounting_file} & {budget_file}")

        st.divider()
        st.header("2. Primary Key Mapping")
        st.info("We've auto-detected the key column (ORS) to perform the matching. Please map Amount and other columns in the section below.")

        # Heuristics for auto-selection
        ors_keywords = ['ors', 'obligation', 'ref', 'reference', 'control']
        amt_keywords = ['gross', 'amount', 'amt', 'total', 'cost', 'net']

        c1, c2 = st.columns(2)
        
        # Accounting Mapping
        with c1:
            st.subheader("Accounting File")
            with st.expander("🔍 View/Browse Data"):
                st.dataframe(df_acc)
            
            acc_ors_idx = find_best_match(df_acc.columns, ors_keywords)
            acc_ors_col = st.selectbox("Select ORS Column (ID)", df_acc.columns, index=acc_ors_idx, key="acc_ors")

            acc_amt_idx = find_best_match(df_acc.columns, amt_keywords)
            acc_amt_col = st.selectbox("Select Amount Column", ["-- None --"] + list(df_acc.columns), index=acc_amt_idx + 1 if acc_amt_idx != 0 else 0, key="acc_amt")

        # Budget Mapping
        with c2:
            st.subheader("Budget File")
            with st.expander("🔍 View/Browse Data"):
                st.dataframe(df_bud)

            bud_ors_idx = find_best_match(df_bud.columns, ors_keywords)
            bud_ors_col = st.selectbox("Select ORS Column (ID)", df_bud.columns, index=bud_ors_idx, key="bud_ors")
            
            bud_amt_idx = find_best_match(df_bud.columns, amt_keywords)
            bud_amt_col = st.selectbox("Select Amount Column", ["-- None --"] + list(df_bud.columns), index=bud_amt_idx + 1 if bud_amt_idx != 0 else 0, key="bud_amt")

        # --- Dynamic Column Mapping ---
        st.divider()
        st.subheader("🔗 Additional Columns to Compare")
        st.info("Below are the columns from the file with fewer columns. Please map them to the corresponding columns in the other file.")

        # Determine which file has fewer columns to be the "Driver"
        # If equal, default to Accounting as driver
        if len(df_acc.columns) <= len(df_bud.columns):
            driver_name = "Accounting"
            target_name = "Budget"
            # Exclude ORS, Amount, and Unnamed columns
            exclude_cols = [acc_ors_col]
            if acc_amt_col != "-- None --": exclude_cols.append(acc_amt_col)
            
            driver_cols = [c for c in df_acc.columns if c not in exclude_cols and not str(c).startswith('Unnamed:') and str(c) != 'Column1']
            target_cols = [c for c in df_bud.columns if not str(c).startswith('Unnamed:') and str(c) != 'Column1']
            is_acc_driver = True
        else:
            driver_name = "Budget"
            target_name = "Accounting"
            exclude_cols = [bud_ors_col]
            if bud_amt_col != "-- None --": exclude_cols.append(bud_amt_col)
            
            driver_cols = [c for c in df_bud.columns if c not in exclude_cols and not str(c).startswith('Unnamed:') and str(c) != 'Column1']
            target_cols = [c for c in df_acc.columns if not str(c).startswith('Unnamed:') and str(c) != 'Column1']
            is_acc_driver = False

        st.markdown(f"**Mapping Source:** `{driver_name} File` ({len(driver_cols)} columns to map)")
        
        # Toggle for Auto-Matching
        auto_match = st.checkbox("✨ Auto-match columns by name", value=True, help="If unchecked, all columns will default to 'Skip' so you can manually select only the ones you want.")

        cols_to_compare = []
        
        # Container for the mapping grid
        with st.container():
            st.markdown("---")
            # Header
            h1, h2, h3 = st.columns([2, 0.5, 2])
            h1.markdown(f"**{driver_name} Column**")
            h2.markdown("➡️")
            h3.markdown(f"**{target_name} Column**")
            
            st.markdown("---")

            # Dynamic Rows
            for i, col in enumerate(driver_cols):
                c1, c2, c3 = st.columns([2, 0.5, 2])
                
                with c1:
                    st.write(f"📄 {col}")
                
                with c2:
                    st.write("➡️")

                with c3:
                    # Default to Skip
                    match_idx = 0 
                    
                    if auto_match:
                        # Try to find best match automatically
                        norm_col = normalize_col(col)
                        
                        # Check for exact normalized match
                        for idx, t_col in enumerate(target_cols):
                            if normalize_col(t_col) == norm_col:
                                match_idx = idx + 1
                                break
                    
                    selected_target = st.selectbox(
                        f"Map {col}", 
                        ["-- Skip --"] + target_cols,
                        index=match_idx,
                        key=f"map_{i}_{col}",
                        label_visibility="collapsed"
                    )

                    if selected_target != "-- Skip --":
                        if is_acc_driver:
                            cols_to_compare.append({
                                "acc_col": col,
                                "bud_col": selected_target,
                                "norm": normalize_col(col)
                            })
                        else:
                            cols_to_compare.append({
                                "acc_col": selected_target,
                                "bud_col": col,
                                "norm": normalize_col(col)
                            })

        st.divider()
        
        if st.button("Run Reconciliation", type="primary"):
            st.header("3. Results")
            
            # Use selected amount columns (or None if skipped)
            final_acc_amt_col = acc_amt_col if acc_amt_col != "-- None --" else None
            final_bud_amt_col = bud_amt_col if bud_amt_col != "-- None --" else None

            # Overview Section
            st.subheader("📊 Dataset Overview")
            c_ov1, c_ov2 = st.columns(2)
            
            # Calculate unique ORS counts
            acc_unique_ors = df_acc[acc_ors_col].nunique()
            bud_unique_ors = df_bud[bud_ors_col].nunique()
            
            c_ov1.metric("Total Accounting Records (Unique ORS)", f"{acc_unique_ors:,}", help=f"Unique ORS numbers found in {len(df_acc):,} total rows")
            c_ov2.metric("Total Budget Records (Unique ORS)", f"{bud_unique_ors:,}", help=f"Unique ORS numbers found in {len(df_bud):,} total rows")
            st.divider()

            try:
                with st.spinner("Processing reconciliation..."):
                    try:
                        merged = reconcile_data(
                            df_acc, 
                            df_bud, 
                            acc_ors_col, 
                            bud_ors_col, 
                            final_acc_amt_col, 
                            final_bud_amt_col, 
                            cols_to_compare
                        )
                    except Exception as e:
                        st.error(f"Error during reconciliation: {str(e)}")
                        st.stop()

                # Summary Metrics
                # Updated to handle dynamic Status strings (e.g., "Payee Mismatch", "Amount Mismatch, Payee Mismatch")
                
                mask_fully_matched = merged['Status'] == 'Fully Matched'
                mask_missing_bud = merged['Status'] == 'Missing in Budget'
                mask_missing_acc = merged['Status'] == 'Missing in Accounting'
                
                # Check for Amount Mismatch (substring search)
                mask_amt_mismatch = merged['Status'].str.contains('Amount Mismatch', case=False, na=False)
                
                # Check for Data Mismatch (using the boolean flag from core)
                # Note: 'Has_Data_Mismatch' is TRUE for both Amount and Text mismatches
                mask_data_mismatch = merged['Has_Data_Mismatch']
                
                # Metric Rows
                row1_cols = st.columns(3)
                row1_cols[0].metric("✅ Fully Matched", int(mask_fully_matched.sum()))
                row1_cols[1].metric("❌ Missing in Budget", int(mask_missing_bud.sum()))
                row1_cols[2].metric("❌ Missing in Accounting", int(mask_missing_acc.sum()))

                # Consolidated Mismatch Metric (includes Amount & Data)
                row2_cols = st.columns(1)
                row2_cols[0].metric("⚠️ Data Mismatch", int(mask_data_mismatch.sum()), help="Includes both Amount and Text (e.g., Payee) mismatches")

                # --- High-Level Mismatch Drivers (Dashboard) ---
                if mask_data_mismatch.any():
                    # Get all rows with ANY mismatch (excluding missing)
                    all_mismatch_mask = mask_data_mismatch
                    all_mismatch_df = merged[all_mismatch_mask]
                    
                    if not all_mismatch_df.empty:
                        total_counts = {}
                        for reasons in all_mismatch_df['Data_Mismatches']:
                            for r in reasons:
                                if "Amount:" in r: continue # Skip amount reasons here, focus on data columns
                                col = r.split(':')[0]
                                total_counts[col] = total_counts.get(col, 0) + 1
                        
                        if total_counts:
                            st.info("💡 **Where are the mismatches happening?**")
                            # Sort by count descending
                            sorted_reasons = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
                            
                            # Display as a horizontal list of pills/metrics
                            m_cols = st.columns(len(sorted_reasons))
                            for i, (col_name, count) in enumerate(sorted_reasons):
                                # Use limited columns to avoid squeeze
                                if i < 5:
                                    m_cols[i].metric(f"Differs in {col_name}", count)

                # Define the specific columns to show based on user request:
                # Strictly: ORS Number (renamed from Clean_ORS)
                # AND Mismatch Reasons (only where applicable)
                
                def get_filtered_columns(df, include_reasons=False):
                    # We use Clean_ORS as the unified Primary Key
                    cols = ['Clean_ORS']
                    
                    # Add Mismatch Reasons only if requested and exists
                    if include_reasons and 'Mismatch_Reasons' in df.columns:
                        cols.append('Mismatch_Reasons')
                        
                    # Filter to keep only existing columns
                    return [c for c in cols if c in df.columns]

                # Base columns (ORS only)
                basic_display_cols = get_filtered_columns(merged, include_reasons=False)
                # Detailed columns (ORS + Reasons)
                detailed_display_cols = get_filtered_columns(merged, include_reasons=True)
                
                # Detailed Views
                st.subheader("Detailed Breakdown")
                
                tabs = st.tabs(["✅ Fully Matched", "⚠️ Amount Issues", "⚠️ Data Issues", "❌ Missing"])
                
                with tabs[0]:
                    # Rename for display
                    st.dataframe(
                        merged[merged['Status'] == 'Fully Matched'][basic_display_cols]
                        .rename(columns={'Clean_ORS': 'ORS Number'})
                    )
                
                with tabs[1]:
                    st.caption("Records with Amount differences")
                    # Filter: Status contains "Amount Mismatch"
                    mask = merged['Status'].str.contains('Amount Mismatch', case=False, na=False)
                    st.dataframe(
                        merged[mask][detailed_display_cols]
                        .rename(columns={'Clean_ORS': 'ORS Number'})
                    )
                
                with tabs[2]:
                    st.caption("Records where Amounts match, but other columns differ")
                    # Filter: Has Data Mismatch (boolean flag)
                    # Note: This tab might overlap with Amount Issues if both mismatch, which is fine/helpful
                    mask = merged['Has_Data_Mismatch']
                    df_mismatch = merged[mask]

                    # --- Mismatch Breakdown ---
                    if not df_mismatch.empty:
                        mismatch_counts = {}
                        for reasons in df_mismatch['Data_Mismatches']:
                            for r in reasons:
                                if "Amount:" in r: 
                                    continue
                                # Extract column name (everything before the first colon)
                                col = r.split(':')[0]
                                mismatch_counts[col] = mismatch_counts.get(col, 0) + 1
                        
                        if mismatch_counts:
                            st.markdown("### 📉 Mismatch Breakdown")
                            # Convert to DataFrame for nice display
                            breakdown_df = pd.DataFrame(list(mismatch_counts.items()), columns=['Column', 'Count'])
                            breakdown_df = breakdown_df.sort_values('Count', ascending=False)
                            
                            # Display as metrics
                            # Cap at 4 columns to prevent layout issues
                            num_cols = min(len(breakdown_df), 4)
                            cols = st.columns(num_cols)
                            
                            for idx, row in breakdown_df.reset_index(drop=True).iterrows():
                                # Prevent too many columns if many fields mismatch
                                if idx < 4:
                                    with cols[idx]:
                                        st.metric(f"Differs in {row['Column']}", row['Count'])
                                else:
                                    # If more than 4, just show in text
                                    if idx == 4: st.write("**Other Mismatches:**")
                                    st.write(f"- {row['Column']}: {row['Count']}")
                            
                            st.divider()

                            # --- Interactive Filtering ---
                            filter_cols = ["All"] + list(mismatch_counts.keys())
                            selected_filter = st.selectbox("🔍 Filter by Mismatch Type:", filter_cols)

                            if selected_filter != "All":
                                # Filter rows where the Mismatch Reasons contain the selected column
                                # We check if any of the reason strings start with "SelectedColumn:"
                                def has_specific_mismatch(reasons):
                                    for r in reasons:
                                        if r.startswith(f"{selected_filter}:"):
                                            return True
                                    return False
                                
                                df_mismatch = df_mismatch[df_mismatch['Data_Mismatches'].apply(has_specific_mismatch)]
                                st.caption(f"Showing {len(df_mismatch)} rows with mismatches in **{selected_filter}**")

                    st.dataframe(
                        df_mismatch[detailed_display_cols]
                        .rename(columns={'Clean_ORS': 'ORS Number'})
                    )
                
                with tabs[3]:
                    mask = merged['Status'].isin(['Missing in Budget', 'Missing in Accounting'])
                    st.dataframe(
                        merged[mask][basic_display_cols]
                        .rename(columns={'Clean_ORS': 'ORS Number'})
                    )

                # Download Report
                st.divider()
                st.subheader("Download Report")
                
                output = io.BytesIO()
                try:
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Helper to prepare full dataset (excluding internal columns)
                        def get_full_export_df(df):
                            # User Request: Only ORS, Mismatch_Reasons, and Status
                            target_cols = ['Clean_ORS', 'Mismatch_Reasons', 'Status']
                            
                            # Filter to existing columns (if Mismatch_Reasons is missing, it won't crash)
                            final_cols = [c for c in target_cols if c in df.columns]
                            
                            return df[final_cols].rename(columns={'Clean_ORS': 'ORS Number'})

                        # Helper to write sheet
                        def write_sheet(df, sheet_name, include_all_cols=False):
                            if not df.empty:
                                if include_all_cols:
                                    # Use full dataset for detailed review
                                    export_df = get_full_export_df(df)
                                else:
                                    # Use basic simplified view (ORS only)
                                    cols = ['Clean_ORS']
                                    export_df = df[cols].rename(columns={'Clean_ORS': 'ORS Number'})
                                
                                export_df.to_excel(writer, index=False, sheet_name=sheet_name)

                        # Full Reconciliation (Detailed - All Cols)
                        write_sheet(merged, 'Full Reconciliation', include_all_cols=True)
                        
                        # Specialized sheets
                        # Fully Matched -> Basic (User preference: Keep it simple?)
                        # Actually, user said "A report where discrepancies are noted should be ALL include"
                        # So Fully Matched can stay simple or be full. Let's keep it simple for now unless requested.
                        write_sheet(merged[merged['Status'] == 'Fully Matched'], 'Fully Matched', include_all_cols=False)
                        
                        # Mismatches -> Detailed (All Cols)
                        mismatch_mask = merged['Status'].isin(['Amount Mismatch', 'Data Mismatch', 'Amount & Data Mismatch'])
                        write_sheet(merged[mismatch_mask], 'Mismatches', include_all_cols=True)
                        
                        # Missing -> Detailed (All Cols - to see what is missing)
                        # User said "ALL include". Missing records have data from one side.
                        write_sheet(merged[merged['Status'] == 'Missing in Budget'], 'Missing in Budget', include_all_cols=True)
                        write_sheet(merged[merged['Status'] == 'Missing in Accounting'], 'Missing in Accounting', include_all_cols=True)
                        
                        # Unified Discrepancies Sheet (Optional, but helpful)
                        discrepancy_mask = merged['Status'] != 'Fully Matched'
                        write_sheet(merged[discrepancy_mask], 'All Discrepancies', include_all_cols=True)

                except Exception as e:
                    st.error(f"Error generating Excel report: {str(e)}")
                    st.stop()
                
                st.download_button(
                    label="Download Detailed Report (Excel)",
                    data=output.getvalue(),
                    file_name="reconciliation_report_detailed.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)
    
    except Exception as e:
        st.error(f"Error loading files: {e}")

else:
    # Improved feedback for missing files
    st.info("👋 Welcome! To start, please upload your Excel files or select them from the sidebar.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if accounting_file:
            st.success(f"✅ Accounting File Ready\n\n**{accounting_file.name if hasattr(accounting_file, 'name') else accounting_file}**")
        else:
            st.warning("⚠️ Waiting for Accounting File...")
            
    with col2:
        if budget_file:
            st.success(f"✅ Budget File Ready\n\n**{budget_file.name if hasattr(budget_file, 'name') else budget_file}**")
        else:
            st.warning("⚠️ Waiting for Budget File...")
