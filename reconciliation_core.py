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
        for i, col in enumerate[str](columns_norm):
            if keyword_norm in col:
                return i
    return 0

def reconcile_data(df_acc, df_bud, acc_ors_col, bud_ors_col, acc_amt_col, bud_amt_col, cols_to_compare):
    """
    Performs the reconciliation logic on two dataframes.
    
    Args:
        df_acc (pd.DataFrame): Accounting dataframe
        df_bud (pd.DataFrame): Budget dataframe
        acc_ors_col (str): Column name for ORS in accounting
        bud_ors_col (str): Column name for ORS in budget
        acc_amt_col (str): Column name for Amount in accounting
        bud_amt_col (str): Column name for Amount in budget
        cols_to_compare (list): List of dicts with keys 'acc_col', 'bud_col', 'display'
        
    Returns:
        pd.DataFrame: Merged and analyzed dataframe
    """
    # 1. Data Cleaning
    df_acc = df_acc.copy()
    df_bud = df_bud.copy()
    
    df_acc['Clean_ORS'] = df_acc[acc_ors_col].astype(str).str.strip()
    df_bud['Clean_ORS'] = df_bud[bud_ors_col].astype(str).str.strip()

    # Filter out invalid ORS (empty, nan, artifacts) to prevent ghost rows
    invalid_ors = ['nan', 'none', '', 'nat']
    
    # Also filter out if the value matches the column name itself (common header artifact)
    acc_header_val = str(acc_ors_col).strip().lower()
    bud_header_val = str(bud_ors_col).strip().lower()

    # Apply filters
    df_acc = df_acc[~df_acc['Clean_ORS'].str.lower().isin(invalid_ors)]
    df_acc = df_acc[df_acc['Clean_ORS'].str.lower() != acc_header_val]

    df_bud = df_bud[~df_bud['Clean_ORS'].str.lower().isin(invalid_ors)]
    df_bud = df_bud[df_bud['Clean_ORS'].str.lower() != bud_header_val]

    # Handle optional Amount columns
    has_amount = acc_amt_col is not None and bud_amt_col is not None
    
    if has_amount:
        df_acc['Clean_Amount'] = df_acc[acc_amt_col].apply(clean_currency)
        df_bud['Clean_Amount'] = df_bud[bud_amt_col].apply(clean_currency)
        # Create a rounded amount column for safe matching
        df_acc['Match_Amount'] = df_acc['Clean_Amount'].round(2)
        df_bud['Match_Amount'] = df_bud['Clean_Amount'].round(2)
    else:
        # Create dummy columns to satisfy schema if needed, or handle logic branching
        df_acc['Clean_Amount'] = 0.0
        df_bud['Clean_Amount'] = 0.0
        df_acc['Match_Amount'] = 0.0
        df_bud['Match_Amount'] = 0.0

    # 2. Perform Matching
    
    # Prep for unique identification
    df_acc['_uid'] = df_acc.index
    df_bud['_uid'] = df_bud.index
    
    matched_acc_ids = set()
    matched_bud_ids = set()
    merged_exact = pd.DataFrame()

    # --- STAGE 1: Exact Match (ORS + Amount) ---
    # Only run if we have amount columns
    if has_amount:
        # Handle duplicates within same ORS+Amount (e.g. two entries of $50 for ORS #1)
        df_acc['Instance_ID_1'] = df_acc.groupby(['Clean_ORS', 'Match_Amount']).cumcount()
        df_bud['Instance_ID_1'] = df_bud.groupby(['Clean_ORS', 'Match_Amount']).cumcount()
        
        merged_exact = pd.merge(
            df_acc,
            df_bud,
            on=['Clean_ORS', 'Match_Amount', 'Instance_ID_1'],
            how='inner',
            suffixes=('_ACC', '_BUD')
        )
        merged_exact['_merge'] = 'both' # Mark as matched
        
        # Identify which rows were matched in Stage 1
        matched_acc_ids = set(merged_exact['_uid_ACC'])
        matched_bud_ids = set(merged_exact['_uid_BUD'])
    
    # --- STAGE 2: Remaining by ORS (Amount Mismatches & Orphans) ---
    # Filter out rows already matched in Stage 1
    rem_acc = df_acc[~df_acc['_uid'].isin(matched_acc_ids)].copy()
    rem_bud = df_bud[~df_bud['_uid'].isin(matched_bud_ids)].copy()
    
    # Handle duplicates for remaining ORS (e.g. leftover rows that didn't match amount)
    rem_acc['Instance_ID_2'] = rem_acc.groupby(['Clean_ORS']).cumcount()
    rem_bud['Instance_ID_2'] = rem_bud.groupby(['Clean_ORS']).cumcount()
    
    merged_rem = pd.merge(
        rem_acc,
        rem_bud,
        on=['Clean_ORS', 'Instance_ID_2'],
        how='outer',
        suffixes=('_ACC', '_BUD'),
        indicator=True
    )
    
    # --- COMBINE RESULTS ---
    # Concatenate Stage 1 and Stage 2
    if not merged_exact.empty:
        merged = pd.concat([merged_exact, merged_rem], ignore_index=True)
    else:
        merged = merged_rem

    # 3. Analysis
    if has_amount:
        merged['Amount_Diff'] = merged['Clean_Amount_ACC'].fillna(0) - merged['Clean_Amount_BUD'].fillna(0)
        merged['Abs_Diff'] = merged['Amount_Diff'].abs()
    else:
        merged['Amount_Diff'] = 0.0
        merged['Abs_Diff'] = 0.0

    tolerance = 0.01

    # Prepare columns for comparison (handling suffixes from merge)
    final_comparison_cols = []
    for pair in cols_to_compare:
        acc_col = pair['acc_col']
        bud_col = pair['bud_col']
        display_name = pair.get('display', acc_col)
        
        # If names are identical, pandas applies suffixes
        if acc_col == bud_col:
            acc_target = f"{acc_col}_ACC"
            bud_target = f"{bud_col}_BUD"
        else:
            acc_target = acc_col
            bud_target = bud_col
        
        final_comparison_cols.append({
            'display': display_name,
            'acc_target': acc_target,
            'bud_target': bud_target
        })

    # Compare Additional Columns
    def check_discrepancies(row):
        if row['_merge'] != 'both':
            return []
        
        mismatches = []
        
        # 1. Check Amount Discrepancy
        if row['Abs_Diff'] > tolerance:
            # Use Clean_Amount as the source of truth for value comparison
            acc_val = row['Clean_Amount_ACC']
            bud_val = row['Clean_Amount_BUD']
            mismatches.append(f"Amount: '{acc_val}' (Acc) vs '{bud_val}' (Bud)")
            
        # 2. Check Mapped Columns Discrepancy
        for item in final_comparison_cols:
            # Safe access to ensure columns exist
            if item['acc_target'] not in row.index or item['bud_target'] not in row.index:
                continue

            acc_val = row[item['acc_target']]
            bud_val = row[item['bud_target']]
            
            # Basic string comparison (normalize for safety)
            val1 = str(acc_val).strip() if pd.notnull(acc_val) else ""
            val2 = str(bud_val).strip() if pd.notnull(bud_val) else ""
            
            # Case-insensitive comparison
            if val1.lower() != val2.lower():
                # Detailed formatted string
                detail = f"{item['display']}: '{val1}' (Acc) vs '{val2}' (Bud)"
                mismatches.append(detail) 
        return mismatches

    merged['Data_Mismatches'] = merged.apply(check_discrepancies, axis=1)
    merged['Has_Data_Mismatch'] = merged['Data_Mismatches'].apply(lambda x: len(x) > 0)
    merged['Mismatch_Reasons'] = merged['Data_Mismatches'].apply(lambda x: " | ".join(x))

    # Categorize
    def categorize(row):
        if row['_merge'] == 'left_only':
            return 'Missing in Budget'
        elif row['_merge'] == 'right_only':
            return 'Missing in Accounting'
        elif row['_merge'] == 'both':
            is_amount_diff = row['Abs_Diff'] > tolerance
            
            # Check for data mismatches other than Amount
            # (Since check_discrepancies adds Amount error to the list, we must filter it out)
            mismatches = row['Data_Mismatches']
            non_amount_mismatches = [m for m in mismatches if not m.startswith("Amount:")]
            is_other_data_diff = len(non_amount_mismatches) > 0
            
            if is_amount_diff and is_other_data_diff:
                return 'Amount & Data Mismatch'
            elif is_amount_diff:
                return 'Amount Mismatch'
            elif is_other_data_diff:
                return 'Data Mismatch'
            else:
                return 'Fully Matched'
        return 'Unknown'

    merged['Status'] = merged.apply(categorize, axis=1)
    
    return merged
