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
        /* --- DAY MODE (Aligned to Design Prompt) --- */
        
        /* GLOBAL TYPOGRAPHY */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        /* 1. Main Background & Text */
        .stApp {
            background-color: #F9FAFB; /* Soft off-white (Gray 50) */
            color: #374151; /* Gray 700 - High contrast but not harsh */
        }
        
        /* 2. Text Hierarchy */
        h1, h2, h3 {
            color: #111827 !important; /* Gray 900 - Near Black */
            font-weight: 600;
        }
        h4, h5, h6 {
            color: #1F2937 !important; /* Gray 800 */
            font-weight: 500;
        }
        p, li, label, .stMarkdown, .stText {
            color: #374151 !important; /* Gray 700 - Reduced eye strain */
            line-height: 1.6; /* Comfortable reading spacing */
        }
        small, .stCaption {
            color: #6B7280 !important; /* Gray 500 - Secondary text */
        }

        /* 3. Sidebar - Distinct but subtle */
        [data-testid="stSidebar"] {
            background-color: #F3F4F6; /* Gray 100 */
            border-right: 1px solid #E5E7EB; /* Gray 200 */
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] .stMarkdown {
             color: #1F2937 !important;
        }
        
        /* 4. Cards & Containers (DataFrames) */
        [data-testid="stDataFrame"] {
            background-color: #F8F9FA !important; /* Very Light Gray */
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); /* Soft shadow */
            padding: 2px;
        }
        [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] span {
            color: #000000 !important; /* Pure Black */
        }

        /* 5. Toolbar (Clean & Minimal) */
        [data-testid="stElementToolbar"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            color: #374151 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        [data-testid="stElementToolbar"] svg {
            fill: #6B7280 !important; /* Gray 500 */
        }
        [data-testid="stElementToolbar"]:hover {
            background-color: #F9FAFB !important;
        }
        [data-testid="stElementToolbar"]:hover svg {
            fill: #111827 !important; /* Darker on hover */
        }

        /* 6. Inputs (Selectbox, Text Input) */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #111827 !important;
            border: 1px solid #D1D5DB !important; /* Gray 300 */
            border-radius: 6px;
        }
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
             border-color: #6366F1 !important; /* Indigo 500 - Subtle Accent */
             box-shadow: 0 0 0 1px #6366F1 !important;
        }

        /* 7. Dropdown Menu Options */
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border-radius: 6px;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            color: #374151 !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
            background-color: #F3F4F6 !important;
        }

        /* 8. File Uploader - "Surface" Style */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF;
            border: 1px dashed #D1D5DB; /* Gray 300 */
            padding: 16px;
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #9CA3AF; /* Gray 400 */
            background-color: #F9FAFB;
        }
        [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
            color: #4B5563 !important; /* Gray 600 */
        }

        /* 9. Buttons - Soft & Professional */
        button:not([data-testid="baseButton-primary"]) {
            background-color: #FFFFFF !important;
            color: #374151 !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 6px;
            font-weight: 500;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            transition: all 0.15s ease;
        }
        button:not([data-testid="baseButton-primary"]):hover {
            background-color: #F9FAFB !important;
            border-color: #9CA3AF !important;
            color: #111827 !important;
            box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.05);
        }

        /* 10. Tooltips */
        div[role="tooltip"] {
            background-color: #FFFFFF !important;
            color: #1F2937 !important;
            border: 1px solid #E5E7EB !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 6px;
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
    key="acc_file_v5"
)
uploaded_bud = st.sidebar.file_uploader(
    "Budget File", 
    type=["xlsx", "xls", "csv", "xlsm", "xlsb"], 
    key="bud_file_v5"
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
            try:
                return pd.read_excel(file, engine="calamine")
            except:
                # Fallback to default (let pandas decide)
                if hasattr(file, 'seek'): file.seek(0)
                return pd.read_excel(file)

    # 3. Strict .xlsx handling (Use openpyxl ONLY)
    if filename.endswith(('.xlsx', '.xlsm', '.xltx', '.xltm')):
        try:
            return pd.read_excel(file, engine="openpyxl")
        except Exception:
            if hasattr(file, 'seek'): file.seek(0)
            try:
                return pd.read_excel(file, engine="calamine")
            except:
                # Fallback to default
                if hasattr(file, 'seek'): file.seek(0)
                return pd.read_excel(file)

    # 4. Fallback for unknown extensions
    try:
        return pd.read_excel(file, engine="calamine") 
    except:
        if hasattr(file, 'seek'): file.seek(0)
        return pd.read_excel(file) 

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
        ors_keywords = ['ors', 'obligation', 'ref', 'reference', 'control', 'mfo', 'pap']
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
        
        # --- SESSION STATE MANAGEMENT ---
        # Initialize session state for reconciliation results if not present
        if 'reconciliation_result' not in st.session_state:
            st.session_state.reconciliation_result = None
            st.session_state.dropped_records = None
            st.session_state.acc_unique_ors = 0
            st.session_state.bud_unique_ors = 0
            st.session_state.has_run = False

        # Run Reconciliation Button
        if st.button("Run Reconciliation", type="primary"):
            st.session_state.has_run = True
            
            # Use selected amount columns (or None if skipped)
            final_acc_amt_col = acc_amt_col if acc_amt_col != "-- None --" else None
            final_bud_amt_col = bud_amt_col if bud_amt_col != "-- None --" else None

            # Calculate unique ORS counts
            st.session_state.acc_unique_ors = df_acc[acc_ors_col].nunique()
            st.session_state.bud_unique_ors = df_bud[bud_ors_col].nunique()

            try:
                with st.spinner("Processing reconciliation..."):
                    try:
                        temp_result = reconcile_data(
                            df_acc, 
                            df_bud, 
                            acc_ors_col, 
                            bud_ors_col, 
                            final_acc_amt_col, 
                            final_bud_amt_col, 
                            cols_to_compare
                        )
                        # st.write(f"Debug: Returned {len(temp_result)} values") 
                        merged, dropped = temp_result
                        
                        st.session_state.reconciliation_result = merged
                        st.session_state.dropped_records = dropped
                    except Exception as e:
                        import traceback
                        st.error(f"Error during reconciliation: {str(e)}")
                        st.code(traceback.format_exc())
                        st.session_state.reconciliation_result = None
                        st.session_state.has_run = False
                        st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.reconciliation_result = None
                st.session_state.has_run = False

        # --- RESULTS DISPLAY ---
        # Check if results exist in session state
        if st.session_state.has_run and st.session_state.reconciliation_result is not None:
            st.header("3. Results")
            
            merged = st.session_state.reconciliation_result

            # Overview Section
            st.subheader("📊 Dataset Overview")
            c_ov1, c_ov2, c_ov3 = st.columns(3)
            
            c_ov1.metric("Total Accounting Records (Unique ORS)", f"{st.session_state.acc_unique_ors:,}")
            c_ov2.metric("Total Budget Records (Unique ORS)", f"{st.session_state.bud_unique_ors:,}")
            
            dropped_count = len(st.session_state.dropped_records) if st.session_state.dropped_records is not None else 0
            c_ov3.metric("Dropped (Blank Data)", f"{dropped_count:,}", help="Rows excluded due to blank values in mapped columns")
            
            st.divider()

            try:
                # Summary Metrics
                # Updated to handle dynamic Status strings (e.g., "Payee Mismatch", "Amount Mismatch, Payee Mismatch")
                
                mask_fully_matched = merged['Status'] == 'Fully Matched'
                mask_missing_bud = merged['Status'] == 'Missing in Budget'
                mask_missing_acc = merged['Status'] == 'Missing in Accounting'
                
                # Consolidated Mismatch Mask
                # Any row that is NOT Fully Matched and NOT Missing is considered a "Data Mismatch"
                # This covers: "Amount Mismatch", "Payee Mismatch", "Amount Mismatch, Payee Mismatch", etc.
                mask_data_mismatch = ~(mask_fully_matched | mask_missing_bud | mask_missing_acc)
                
                # Metric Rows
                row1_cols = st.columns(3)
                row1_cols[0].metric("✅ Fully Matched", int(mask_fully_matched.sum()))
                row1_cols[1].metric("❌ Missing in Budget", int(mask_missing_bud.sum()))
                row1_cols[2].metric("❌ Missing in Accounting", int(mask_missing_acc.sum()))

                # Consolidated Mismatch Metric
                row2_cols = st.columns(1)
                row2_cols[0].metric("⚠️ Data Mismatch", int(mask_data_mismatch.sum()), help="Includes ORS matches with differences in Amount or Text (e.g., Payee)")

                # --- High-Level Mismatch Drivers (Dashboard) ---
                if mask_data_mismatch.any():
                    # Get all rows with ANY mismatch
                    all_mismatch_df = merged[mask_data_mismatch]
                    
                    if not all_mismatch_df.empty and 'Mismatch_Reasons' in all_mismatch_df.columns:
                        total_counts = {}
                        
                        # Helper to parse reasons safely
                        def parse_reasons(val):
                            if isinstance(val, list): return val
                            if isinstance(val, str): return val.split(', ')
                            return []

                        for reasons in all_mismatch_df['Mismatch_Reasons'].apply(parse_reasons):
                            for r in reasons:
                                # Count specific mismatch types
                                if "Mismatch" in r:
                                    col = r.replace(" Mismatch", "")
                                    total_counts[col] = total_counts.get(col, 0) + 1
                        
                        if total_counts:
                            st.info("💡 **Where are the mismatches happening?**")
                            # Sort by count descending
                            sorted_reasons = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
                            
                            # Display as a horizontal list of pills/metrics
                            num_metrics = min(len(sorted_reasons), 5)
                            if num_metrics > 0:
                                m_cols = st.columns(num_metrics)
                                for i in range(num_metrics):
                                    col_name, count = sorted_reasons[i]
                                    m_cols[i].metric(f"{col_name}", count)

                # Define the specific columns to show based on user request:
                # Strictly: ORS Number (renamed from Clean_ORS)
                # AND Mismatch Reasons (only where applicable)
                
                def get_filtered_columns(df, include_reasons=False):
                    # We use Clean_ORS as the unified Primary Key
                    cols = ['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD']
                    
                    # Add Mismatch Reasons only if requested and exists
                    if include_reasons and 'Mismatch_Reasons' in df.columns:
                        cols.append('Mismatch_Reasons')
                    
                    # Add Status if requested
                    if include_reasons and 'Status' in df.columns:
                        cols.append('Status')
                        
                    # Filter to keep only existing columns
                    return [c for c in cols if c in df.columns]

                # Base columns (ORS + Amounts)
                basic_display_cols = get_filtered_columns(merged, include_reasons=False)
                # Detailed columns (ORS + Amounts + Reasons + Status)
                detailed_display_cols = get_filtered_columns(merged, include_reasons=True)
                
                # Column Renaming Map
                col_rename_map = {
                    'Clean_ORS': 'ORS Number',
                    'Clean_Amount_ACC': 'Amount (Accounting)',
                    'Clean_Amount_BUD': 'Amount (Budget)'
                }

                # Detailed Views
                st.subheader("Detailed Breakdown")
                
                # Consolidated Tabs: Fully Matched | Data Mismatches | Missing in Accounting | Missing in Budget | Dropped Records
                tabs = st.tabs(["✅ Fully Matched", "⚠️ Data Mismatches", "❌ Missing in Accounting", "❌ Missing in Budget", "🗑️ Dropped Records"])
                
                with tabs[0]:
                    # Rename for display
                    st.dataframe(
                        merged[mask_fully_matched][basic_display_cols]
                        .rename(columns=col_rename_map)
                    )
                
                with tabs[1]:
                    st.caption("Records where ORS matches but Amount or Data differs")
                    
                    df_mismatch = merged[mask_data_mismatch]

                    if not df_mismatch.empty:
                        # --- Interactive Filtering ---
                        # Extract all unique mismatch types for the filter (from Status column)
                        unique_types = set()
                        
                        if 'Status' in df_mismatch.columns:
                            # Split status strings by ", " to get individual categories
                            # e.g. "Amount Mismatch, Payee Mismatch" -> ["Amount Mismatch", "Payee Mismatch"]
                            for status_str in df_mismatch['Status'].dropna().astype(str):
                                parts = [p.strip() for p in status_str.split(',')]
                                for part in parts:
                                    if part: unique_types.add(part)
                        
                        filter_opts = ["All"] + sorted(list(unique_types))
                        selected_filter = st.selectbox("🔍 Filter by Mismatch Type:", filter_opts, key="mismatch_filter_select")

                        if selected_filter != "All":
                            # Filter rows where Status contains the selected type
                            # Using str.contains is robust and handles the "contains" logic naturally
                            # (e.g., selecting "Amount Mismatch" matches "Amount Mismatch, Payee Mismatch")
                            mask_filter = df_mismatch['Status'].astype(str).str.contains(selected_filter, regex=False, case=False)
                            df_mismatch = df_mismatch[mask_filter]
                            
                            st.caption(f"Showing {len(df_mismatch)} rows matching **{selected_filter}**")

                    st.dataframe(
                        df_mismatch[detailed_display_cols]
                        .rename(columns=col_rename_map)
                    )
                
                with tabs[2]:
                    # Missing in Accounting
                    st.caption("Records present in Budget but missing in Accounting")
                    st.dataframe(
                        merged[mask_missing_acc][basic_display_cols]
                        .rename(columns=col_rename_map)
                    )
                    
                with tabs[3]:
                    # Missing in Budget
                    st.caption("Records present in Accounting but missing in Budget")
                    st.dataframe(
                        merged[mask_missing_bud][basic_display_cols]
                        .rename(columns=col_rename_map)
                    )

                with tabs[4]:
                    # Dropped Records
                    st.caption("Records excluded due to Zero Amount or Blank Data in mapped columns")
                    if st.session_state.dropped_records is not None and not st.session_state.dropped_records.empty:
                        st.dataframe(st.session_state.dropped_records)
                    else:
                        st.info("No records were dropped.")

                # Download Report
                st.divider()
                st.subheader("Download Report")
                
                output = io.BytesIO()
                try:
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Helper to prepare full dataset (excluding internal columns)
                        def get_full_export_df(df):
                            # User Request: ORS, Amounts, Mismatch_Reasons, and Status
                            target_cols = ['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD', 'Mismatch_Reasons', 'Status']
                            
                            # Filter to existing columns
                            final_cols = [c for c in target_cols if c in df.columns]
                            
                            return df[final_cols].rename(columns=col_rename_map)

                        # Helper to write sheet
                        def write_sheet(df, sheet_name, include_all_cols=False):
                            if not df.empty:
                                if include_all_cols:
                                    # Use full dataset for detailed review
                                    export_df = get_full_export_df(df)
                                else:
                                    # Use basic simplified view (ORS + Amounts)
                                    cols = ['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD']
                                    export_df = df[cols].rename(columns=col_rename_map)
                                
                                export_df.to_excel(writer, index=False, sheet_name=sheet_name)

                        # Full Reconciliation (Detailed - All Cols)
                        write_sheet(merged, 'Full Reconciliation', include_all_cols=True)
                        
                        # Fully Matched
                        write_sheet(merged[mask_fully_matched], 'Fully Matched', include_all_cols=False)
                        
                        # Data Mismatches (Consolidated)
                        write_sheet(merged[mask_data_mismatch], 'Data Mismatches', include_all_cols=True)
                        
                        # Missing
                        write_sheet(merged[mask_missing_bud], 'Missing in Budget', include_all_cols=True)
                        write_sheet(merged[mask_missing_acc], 'Missing in Accounting', include_all_cols=True)
                        
                        # Dropped Records (Blank Data)
                        if st.session_state.dropped_records is not None and not st.session_state.dropped_records.empty:
                            st.session_state.dropped_records.to_excel(writer, index=False, sheet_name='Dropped - Blank Entries')
                        
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
