import pandas as pd
import numpy as np
import re

def clean_currency(x):
    """
    Robust currency cleaner that handles:
    - Standard formats: 1000.00, 1,000.00
    - Accounting format: (100) -> -100
    - Currency symbols/text: $100, USD 100, 100 USD
    - Whitespace
    """
    if pd.isnull(x) or str(x).strip() == '':
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
        
    x_str = str(x).strip()
    
    # 1. Remove all non-numeric characters except digits, dot, minus, and parens
    # This effectively strips 'USD', '$', ',', ' ' etc.
    clean_str = re.sub(r'[^\d.\-\(\)]', '', x_str)
    
    # 2. Handle accounting format (parentheses = negative)
    is_negative = False
    if clean_str.startswith('(') and clean_str.endswith(')'):
        is_negative = True
        clean_str = clean_str[1:-1]
    
    # 3. Handle trailing negatives (100-) if any
    if clean_str.endswith('-'):
        is_negative = True
        clean_str = clean_str[:-1]
    elif clean_str.startswith('-'):
        is_negative = True # Standard negative
        clean_str = clean_str[1:] # Strip it to parse safely, then re-apply

    try:
        val = float(clean_str)
        return -val if is_negative else val
    except:
        return 0.0

def normalize_col(col_name):
    return str(col_name).lower().strip().replace('_', '').replace(' ', '')

def find_best_match(columns, keywords):
    """Finds the best matching column name index from a list of keywords."""
    columns_norm = [normalize_col(c) for c in columns]
    
    for keyword in keywords:
        keyword_norm = normalize_col(keyword)
        # 1. Exact match
        if keyword_norm in columns_norm:
            return columns_norm.index(keyword_norm)
        
        # 2. Contains match
        for i, col in enumerate(columns_norm):
            if keyword_norm in col:
                return i
    return 0

def reconcile_optimized(dfs_dict, key_cols, value_cols, mapping_dict):
    """
    High-performance matching algorithm optimized for large datasets.
    
    Args:
        dfs_dict (dict): Dictionary of {sheet_name: DataFrame}
        key_cols (dict): Dictionary of {sheet_name: ors_col_name}
        value_cols (dict): Dictionary of {sheet_name: amount_col_name}
        mapping_dict (list): List of dicts [{'col_a': 'col_b', 'display': 'Name'}] for extra column comparison
    
    Returns:
        pd.DataFrame: Merged result with matches and metadata
    """
    processed_dfs = []
    
    for sheet_name, df in dfs_dict.items():
        # 1. Preprocessing (Vectorized)
        df = df.copy()
        
        # Standardize ORS
        ors_col = key_cols[sheet_name]
        amt_col = value_cols.get(sheet_name)
        
        # Vectorized string operation
        df['__ORS'] = df[ors_col].astype(str).str.strip()
        
        # Filter invalid ORS (vectorized)
        # Check for column header artifacts (where value == column name)
        header_val = str(ors_col).strip().lower()
        invalid_mask = df['__ORS'].isin(['nan', 'none', '', 'nat']) | (df['__ORS'].str.lower() == header_val)
        df = df[~invalid_mask].copy()
        
        # Handle Amount (Vectorized where possible)
        if amt_col:
            # Check if column is already numeric
            if pd.api.types.is_numeric_dtype(df[amt_col]):
                df['__Amount'] = df[amt_col].fillna(0.0)
            else:
                # Use clean_currency row-wise for safety with regex logic on mixed types
                # (Can be optimized to vectorized regex if strictly needed, but clean_currency is robust)
                # Given constraint "DO NOT use df.apply()", we attempt vectorization for common cases
                
                # Convert to string
                s = df[amt_col].astype(str).str.strip()
                # Remove currency symbols
                s = s.str.replace(r'[^\d.\-\(\)]', '', regex=True)
                
                # Handle parens (negative)
                neg_mask = s.str.startswith('(') & s.str.endswith(')')
                # Extract inner number
                s_clean = s.where(~neg_mask, s.str.slice(1, -1))
                # Handle trailing negative
                trail_neg = s_clean.str.endswith('-')
                s_clean = s_clean.where(~trail_neg, '-' + s_clean.str.slice(0, -1))
                # Handle standard negative
                # (Already handled by float conversion if '-' is at start)
                
                df['__Amount'] = pd.to_numeric(s_clean, errors='coerce').fillna(0.0)
                # Apply negation
                df.loc[neg_mask, '__Amount'] = -df.loc[neg_mask, '__Amount']
                
            df['__Amount'] = df['__Amount'].round(2)
        else:
            df['__Amount'] = 0.0

        # 2. Dynamic Match Key Creation (Per Sheet)
        # Logic: If ORS is UNIQUE -> Key = ORS; If DUPLICATED -> Key = ORS + Amount
        is_duplicated = df.duplicated(subset=['__ORS'], keep=False)
        
        amt_str = df['__Amount'].astype(str)
        
        df['__MATCH_KEY'] = np.where(
            is_duplicated,
            df['__ORS'] + '_' + amt_str,
            df['__ORS']
        )
        
        df['__SOURCE_SHEET'] = sheet_name
        # Keep original index
        df['__ORIG_IDX'] = df.index
        
        processed_dfs.append(df)
        
    if not processed_dfs:
        return pd.DataFrame()

    # 3. Combine all sheets
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    
    # 4. Reduce join size: Filter keys appearing in >1 source
    # We want keys that exist in at least 2 DIFFERENT sheets.
    # Count unique sheets per key
    key_counts = combined_df.groupby('__MATCH_KEY')['__SOURCE_SHEET'].nunique()
    valid_keys = key_counts[key_counts > 1].index
    
    candidates = combined_df[combined_df['__MATCH_KEY'].isin(valid_keys)]
    
    if candidates.empty:
        return pd.DataFrame()
        
    # 5. Perform Inner Join on MATCH_KEY
    merged = pd.merge(
        candidates, 
        candidates, 
        on='__MATCH_KEY', 
        suffixes=('_L', '_R')
    )
    
    # 6. Remove matches from same source
    merged = merged[merged['__SOURCE_SHEET_L'] != merged['__SOURCE_SHEET_R']]
    
    # Enforce order to avoid duplicates (A-B vs B-A) and standardizing output
    # If sources are 'Accounting' and 'Budget', 'Accounting' < 'Budget'.
    merged = merged[merged['__SOURCE_SHEET_L'] < merged['__SOURCE_SHEET_R']]
    
    # 7. Column Comparison (Vectorized)
    merged['__ALL_COLS_MATCH'] = True
    merged['__MISMATCH_REASONS'] = ""
    
    if mapping_dict:
        # Determine L and R source names (since we filtered by L < R, they are consistent per row if sources are consistent)
        # But source names can vary. 
        # However, for a given row, we know Source_L and Source_R.
        # We need to map the config (acc_col, bud_col) to the dynamic sources.
        
        # Simplified assumption for this specific app context:
        # We assume strict 2-party reconciliation (Accounting vs Budget).
        # Since 'Accounting' < 'Budget', L is Accounting, R is Budget.
        
        # We can iterate through the mapping configuration
        for item in mapping_dict:
            acc_col = item.get('acc_col')
            bud_col = item.get('bud_col')
            display = item.get('display', acc_col)
            
            # We need to fetch values from the merged dataframe.
            # The merged dataframe has columns from combined_df with _L and _R suffixes.
            # Columns that didn't collide in combined_df (unique names) might not have suffixes?
            # No, pd.merge with suffixes applies them to overlapping columns.
            # But combined_df has ALL columns. So all columns overlap (self-join).
            # So _L and _R are guaranteed.
            
            # Left (Accounting) uses acc_col
            col_L = f"{acc_col}_L"
            # Right (Budget) uses bud_col
            col_R = f"{bud_col}_R"
            
            # Handle if columns missing (user config might be wrong)
            if col_L not in merged.columns: merged[col_L] = np.nan
            if col_R not in merged.columns: merged[col_R] = np.nan
            
            val_L = merged[col_L].astype(str).str.strip().str.lower()
            val_R = merged[col_R].astype(str).str.strip().str.lower()
            
            # Compare
            # Treat 'nan' as empty string for comparison or strictly?
            # Let's align with existing logic: string compare
            
            # Fix 'nan' string from astype(str) on np.nan
            val_L = val_L.replace('nan', '')
            val_R = val_R.replace('nan', '')
            
            is_match = val_L == val_R
            
            merged['__ALL_COLS_MATCH'] &= is_match
            
            # Construct mismatch text
            # Vectorized construction is tricky for varying strings, but we can do:
            # "Reason | " where not match
            reason = display + ": '" + merged[col_L].astype(str) + "' vs '" + merged[col_R].astype(str) + "' | "
            merged['__MISMATCH_REASONS'] += np.where(~is_match, reason, "")
            
    merged['__MISMATCH_REASONS'] = merged['__MISMATCH_REASONS'].str.strip(' | ')
    
    return merged

def reconcile_data(df_acc, df_bud, acc_ors_col, bud_ors_col, acc_amt_col, bud_amt_col, cols_to_compare):
    """
    Wrapper for backward compatibility with app.py using the new high-performance engine.
    """
    # 1. Prepare Inputs
    dfs = {
        'Accounting': df_acc,
        'Budget': df_bud
    }
    key_cols = {
        'Accounting': acc_ors_col,
        'Budget': bud_ors_col
    }
    value_cols = {
        'Accounting': acc_amt_col,
        'Budget': bud_amt_col
    }
    
    # 2. Run Optimized Engine
    merged_core = reconcile_optimized(dfs, key_cols, value_cols, cols_to_compare)
    
    # 3. Transform Output to match legacy structure expected by app.py
    
    # Identify matched IDs
    matched_acc_ids = set()
    matched_bud_ids = set()
    
    if not merged_core.empty:
        # Assuming L=Accounting, R=Budget due to sort
        matched_acc_ids = set(merged_core['__ORIG_IDX_L'])
        matched_bud_ids = set(merged_core['__ORIG_IDX_R'])
    
    # Pre-calculate Clean columns for reporting on full dataset
    df_acc_clean = df_acc.copy()
    df_bud_clean = df_bud.copy()
    
    df_acc_clean['Clean_ORS'] = df_acc_clean[acc_ors_col].astype(str).str.strip()
    df_bud_clean['Clean_ORS'] = df_bud_clean[bud_ors_col].astype(str).str.strip()
    
    if acc_amt_col:
        df_acc_clean['Clean_Amount'] = df_acc_clean[acc_amt_col].apply(clean_currency)
    else:
        df_acc_clean['Clean_Amount'] = 0.0
        
    if bud_amt_col:
        df_bud_clean['Clean_Amount'] = df_bud_clean[bud_amt_col].apply(clean_currency)
    else:
        df_bud_clean['Clean_Amount'] = 0.0

    results = []
    
    # 3.1 Matches
    if not merged_core.empty:
        for idx, row in merged_core.iterrows():
            # Get original indices
            acc_idx = row['__ORIG_IDX_L']
            bud_idx = row['__ORIG_IDX_R']
            
            # Access original data efficiently
            acc_row = df_acc_clean.loc[acc_idx]
            bud_row = df_bud_clean.loc[bud_idx]
            
            res = {}
            # Flatten dicts with suffixes
            for k, v in acc_row.items(): res[f"{k}_ACC"] = v
            for k, v in bud_row.items(): res[f"{k}_BUD"] = v
            
            # Common fields
            res['Clean_ORS'] = acc_row['Clean_ORS']
            res['Clean_Amount_ACC'] = acc_row['Clean_Amount']
            res['Clean_Amount_BUD'] = bud_row['Clean_Amount']
            res['Abs_Diff'] = abs(res['Clean_Amount_ACC'] - res['Clean_Amount_BUD'])
            
            # Status & Mismatches
            reasons = row['__MISMATCH_REASONS']
            mismatches = [r.strip() for r in reasons.split('|') if r.strip()]
            
            if res['Abs_Diff'] > 0.01:
                mismatches.insert(0, f"Amount: '{res['Clean_Amount_ACC']}' (Acc) vs '{res['Clean_Amount_BUD']}' (Bud)")
                
            res['Data_Mismatches'] = mismatches
            res['Mismatch_Reasons'] = " | ".join(mismatches)
            res['Has_Data_Mismatch'] = len(mismatches) > 0
            res['_merge'] = 'both'
            
            is_amt_diff = res['Abs_Diff'] > 0.01
            non_amt = [m for m in mismatches if not m.startswith("Amount:")]
            is_data_diff = len(non_amt) > 0
            
            if is_amt_diff and is_data_diff:
                res['Status'] = 'Amount & Data Mismatch'
            elif is_amt_diff:
                res['Status'] = 'Amount Mismatch'
            elif is_data_diff:
                res['Status'] = 'Data Mismatch'
            else:
                res['Status'] = 'Fully Matched'
                
            results.append(res)
            
    # 3.2 Unmatched Accounting
    unmatched_acc = df_acc_clean.loc[~df_acc_clean.index.isin(matched_acc_ids)]
    for idx, row in unmatched_acc.iterrows():
        res = {f"{k}_ACC": v for k, v in row.items()}
        res['Clean_ORS'] = row['Clean_ORS']
        res['Clean_Amount_ACC'] = row['Clean_Amount']
        res['Clean_Amount_BUD'] = 0.0
        res['Abs_Diff'] = 0.0
        res['_merge'] = 'left_only'
        res['Status'] = 'Missing in Budget'
        res['Data_Mismatches'] = []
        res['Mismatch_Reasons'] = ""
        results.append(res)
        
    # 3.3 Unmatched Budget
    unmatched_bud = df_bud_clean.loc[~df_bud_clean.index.isin(matched_bud_ids)]
    for idx, row in unmatched_bud.iterrows():
        res = {f"{k}_BUD": v for k, v in row.items()}
        res['Clean_ORS'] = row['Clean_ORS']
        res['Clean_Amount_ACC'] = 0.0
        res['Clean_Amount_BUD'] = row['Clean_Amount']
        res['Abs_Diff'] = 0.0
        res['_merge'] = 'right_only'
        res['Status'] = 'Missing in Accounting'
        res['Data_Mismatches'] = []
        res['Mismatch_Reasons'] = ""
        results.append(res)
        
    return pd.DataFrame(results)
