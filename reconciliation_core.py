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

def clean_id(x):
    """
    Standardizes ID columns (like ORS, MFO) to text:
    - Converts to string
    - Strips whitespace
    - Removes trailing '.0' (Excel float artifact)
    """
    if pd.isnull(x):
        return ""
    s = str(x).strip()
    # Handle Excel float-as-text artifacts: "1234.0" -> "1234"
    if s.endswith('.0'):
        return s[:-2]
    return s

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
        
        # Add metadata columns
        df['__SOURCE_SHEET'] = sheet_name
        df['__ORIG_IDX'] = df.index

        # Standardize ORS
        ors_col = key_cols[sheet_name]
        amt_col = value_cols.get(sheet_name)
        
        # Vectorized string operation with ID cleaning
        # Treat as Text: Convert to string, strip, remove '.0' float artifact
        s = df[ors_col].astype(str).str.strip()
        df['__ORS'] = np.where(s.str.endswith('.0'), s.str[:-2], s)
        
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
        
        processed_dfs.append(df)

    # 2. Match Logic: Two-Pass Approach
    
    # PASS 1: Strict Match (ORS + Amount)
    # This handles duplicates correctly and finds perfect matches
    for df in processed_dfs:
        # Multiply by 100, round, convert to int, then to string. Safe and fast.
        amt_int = (df['__Amount'] * 100).round().astype(int).astype(str)
        df['__MATCH_KEY_STRICT'] = df['__ORS'] + '_' + amt_int
        
    combined_strict = pd.concat(processed_dfs, ignore_index=True)
    
    # Filter keys appearing in >1 source
    key_counts = combined_strict.groupby('__MATCH_KEY_STRICT')['__SOURCE_SHEET'].nunique()
    valid_keys_strict = key_counts[key_counts > 1].index
    
    candidates_strict = combined_strict[combined_strict['__MATCH_KEY_STRICT'].isin(valid_keys_strict)]
    
    merged_strict = pd.DataFrame()
    if not candidates_strict.empty:
        merged_strict = pd.merge(candidates_strict, candidates_strict, on='__MATCH_KEY_STRICT', suffixes=('_L', '_R'))
        # Remove self-matches and enforce order
        merged_strict = merged_strict[merged_strict['__SOURCE_SHEET_L'] != merged_strict['__SOURCE_SHEET_R']]
        merged_strict = merged_strict[merged_strict['__SOURCE_SHEET_L'] < merged_strict['__SOURCE_SHEET_R']]
    
    # Track matched indices to exclude them from Pass 2
    matched_indices_L = set(merged_strict['__ORIG_IDX_L']) if not merged_strict.empty else set()
    matched_indices_R = set(merged_strict['__ORIG_IDX_R']) if not merged_strict.empty else set()
    
    # PASS 2: Loose Match (ORS only) on RESIDUALS
    # This catches "Unique ORS" cases where Amount differs
    
    # Filter residuals
    residuals = []
    for df in processed_dfs:
        sheet_name = df['__SOURCE_SHEET'].iloc[0]
        # Identify which set of matched indices corresponds to this sheet
        # (Assuming 2 sheets, L is first, R is second)
        # Better: check against both sets since we don't know if this sheet was L or R in a specific row
        # Actually, __ORIG_IDX is unique per sheet, but potentially overlapping across sheets?
        # Yes, standard index 0,1,2...
        # So we need to be careful. The merged df has __SOURCE_SHEET_L/R.
        
        # Get matched IDs for THIS sheet
        matched_in_L = merged_strict[merged_strict['__SOURCE_SHEET_L'] == sheet_name]['__ORIG_IDX_L'] if not merged_strict.empty else []
        matched_in_R = merged_strict[merged_strict['__SOURCE_SHEET_R'] == sheet_name]['__ORIG_IDX_R'] if not merged_strict.empty else []
        
        matched_ids = set(matched_in_L).union(set(matched_in_R))
        
        # Keep only unmatched
        res_df = df[~df['__ORIG_IDX'].isin(matched_ids)].copy()
        
        # CRITICAL: Only allow match if ORS is UNIQUE in this residual set?
        # User said: "The ORS match-for UNIQUE ORS"
        # We will filter for uniqueness later or now?
        # If we just merge on ORS, we might get cartesian products for duplicates.
        # We should only keep rows where ORS is unique in this residual DF.
        
        ors_counts = res_df['__ORS'].value_counts()
        unique_ors = ors_counts[ors_counts == 1].index
        res_df_unique = res_df[res_df['__ORS'].isin(unique_ors)]
        
        # We also define the match key as just ORS
        res_df_unique['__MATCH_KEY_LOOSE'] = res_df_unique['__ORS']
        residuals.append(res_df_unique)
        
    combined_loose = pd.concat(residuals, ignore_index=True)
    merged_loose = pd.DataFrame()
    
    if not combined_loose.empty:
        key_counts_loose = combined_loose.groupby('__MATCH_KEY_LOOSE')['__SOURCE_SHEET'].nunique()
        valid_keys_loose = key_counts_loose[key_counts_loose > 1].index
        
        candidates_loose = combined_loose[combined_loose['__MATCH_KEY_LOOSE'].isin(valid_keys_loose)]
        
        if not candidates_loose.empty:
            merged_loose = pd.merge(candidates_loose, candidates_loose, on='__MATCH_KEY_LOOSE', suffixes=('_L', '_R'))
            merged_loose = merged_loose[merged_loose['__SOURCE_SHEET_L'] != merged_loose['__SOURCE_SHEET_R']]
            merged_loose = merged_loose[merged_loose['__SOURCE_SHEET_L'] < merged_loose['__SOURCE_SHEET_R']]
            
    # Combine Matches
    # We need to align columns. 
    # Strict has __MATCH_KEY_STRICT, Loose has __MATCH_KEY_LOOSE.
    # We can drop the key columns or align them.
    
    final_matches = pd.concat([merged_strict, merged_loose], ignore_index=True, sort=False)
    
    if final_matches.empty:
        return pd.DataFrame()
        
    merged = final_matches
    
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
            
            # Clean and normalize Left (treat as text, handle float artifacts)
            s_L = merged[col_L].astype(str).str.strip()
            s_L = np.where(s_L.str.endswith('.0'), s_L.str[:-2], s_L)
            val_L = pd.Series(s_L).str.lower()
            
            # Clean and normalize Right
            s_R = merged[col_R].astype(str).str.strip()
            s_R = np.where(s_R.str.endswith('.0'), s_R.str[:-2], s_R)
            val_R = pd.Series(s_R).str.lower()
            
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
            reason = display + ": '" + merged[col_L].astype(str) + "' (Acc) vs '" + merged[col_R].astype(str) + "' (Bud) | "
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
    
    # Use clean_id to handle text/number artifacts consistently
    df_acc_clean['Clean_ORS'] = df_acc_clean[acc_ors_col].apply(clean_id)
    df_bud_clean['Clean_ORS'] = df_bud_clean[bud_ors_col].apply(clean_id)
    
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
            res['Abs_Diff'] = round(abs(res['Clean_Amount_ACC'] - res['Clean_Amount_BUD']), 2)
            
            # Status & Mismatches
            reasons = row['__MISMATCH_REASONS']
            mismatches = [r.strip() for r in reasons.split('|') if r.strip()]
            
            # Note: Abs_Diff > 0 should not happen with strict matching, but we keep the logic just in case
            if res['Abs_Diff'] > 0:
                mismatches.insert(0, f"Amount: '{res['Clean_Amount_ACC']}' (Acc) vs '{res['Clean_Amount_BUD']}' (Bud)")
                
            res['Data_Mismatches'] = mismatches
            res['Mismatch_Reasons'] = " | ".join(mismatches)
            res['Has_Data_Mismatch'] = len(mismatches) > 0
            res['_merge'] = 'both'
            
            # Simplified Status Logic as per user request
            # 1. Fully Matched
            # 2. Data Mismatch
            #    A. Amount Mismatch
            #    B. Payee Mismatch (or other mapped columns)
            # 3. Missing (Handled in unmatched sections)
            
            status_parts = []
            
            if res['Abs_Diff'] > 0:
                status_parts.append("Amount Mismatch")
                
            if res['Has_Data_Mismatch']:
                # Check for specific column mismatches in reasons
                # The reasons string format is "Display: 'Val1' vs 'Val2'"
                # We can extract the display names
                
                # Check if this is ONLY Amount Mismatch (if we added it to mismatches list above)
                # We added "Amount:" to mismatches list if Abs_Diff > 0
                
                # Filter out the "Amount:" mismatch to check for text mismatches
                text_mismatches = [m for m in mismatches if not m.startswith("Amount:")]
                
                if text_mismatches:
                    # Extract column names from text mismatches
                    # Format: "Column: 'A' vs 'B'"
                    for tm in text_mismatches:
                        col_name = tm.split(':')[0]
                        if f"{col_name} Mismatch" not in status_parts:
                            status_parts.append(f"{col_name} Mismatch")
            
            if not status_parts:
                res['Status'] = 'Fully Matched'
            else:
                res['Status'] = ', '.join(status_parts)
                
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
