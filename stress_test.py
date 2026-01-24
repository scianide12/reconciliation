import pandas as pd
import numpy as np
import time
import os
from reconciliation_core import reconcile_data

def generate_large_data(rows=10000):
    print(f"Generating {rows} rows of data...")
    
    # Common ORS numbers
    ors_ids = [f"ORS-{i:06d}" for i in range(rows)]
    
    # Accounting Data
    df_acc = pd.DataFrame({
        'ORS Number': ors_ids,
        'Amount': np.random.uniform(100, 10000, rows),
        'Payee': [f"Payee {i}" for i in range(rows)],
        'Date': pd.date_range(start='2024-01-01', periods=rows)
    })
    
    # Budget Data (mostly matching, some diffs)
    df_bud = df_acc.copy()
    df_bud.rename(columns={'ORS Number': 'ORS_Ref', 'Amount': 'Gross_Amount'}, inplace=True)
    
    # Introduce discrepancies
    # 1. Modify some amounts
    mask_amt = np.random.choice([True, False], rows, p=[0.05, 0.95])
    df_bud.loc[mask_amt, 'Gross_Amount'] += 10
    
    # 2. Modify some payees (Data mismatch)
    mask_data = np.random.choice([True, False], rows, p=[0.05, 0.95])
    df_bud.loc[mask_data, 'Payee'] = df_bud.loc[mask_data, 'Payee'] + " Inc."
    
    # 3. Drop some rows (Missing in Budget)
    drop_indices = np.random.choice(df_bud.index, int(rows * 0.05), replace=False)
    df_bud = df_bud.drop(drop_indices)
    
    # 4. Add extra rows (Missing in Accounting)
    extra_rows = pd.DataFrame({
        'ORS_Ref': [f"ORS-EXT-{i}" for i in range(100)],
        'Gross_Amount': np.random.uniform(100, 10000, 100),
        'Payee': "Extra Payee",
        'Date': pd.Timestamp('2024-01-01')
    })
    df_bud = pd.concat([df_bud, extra_rows], ignore_index=True)
    
    return df_acc, df_bud

import sys

def run_stress_test():
    # Default to 50000, but allow command line override
    rows = 50000
    if len(sys.argv) > 1:
        try:
            rows = int(sys.argv[1])
        except ValueError:
            pass

    print(f"--- Stress Test Configuration: {rows} rows ---")
    
    # 1. Generate Data
    gen_start = time.time()
    df_acc, df_bud = generate_large_data(rows)
    print(f"✅ Data generation took {time.time() - gen_start:.2f}s")
    
    # 2. Simulate File Write (to measure disk speed impact if we were writing)
    print("Simulating Excel file write (creating test files)...")
    write_start = time.time()
    df_acc.to_excel("stress_acc.xlsx", index=False)
    df_bud.to_excel("stress_bud.xlsx", index=False)
    print(f"✅ Excel write took {time.time() - write_start:.2f}s")
    
    # 3. Simulate File Upload (Read)
    print("Simulating Excel file upload (reading test files)...")
    read_start = time.time()
    df_acc_read = pd.read_excel("stress_acc.xlsx")
    df_bud_read = pd.read_excel("stress_bud.xlsx")
    print(f"✅ Excel read took {time.time() - read_start:.2f}s")
    
    # 4. Reconciliation Process
    print("Starting reconciliation process...")
    process_start = time.time()
    
    cols_to_compare = [{'acc_col': 'Payee', 'bud_col': 'Payee', 'display': 'Payee'}]
    
    try:
        merged = reconcile_data(
            df_acc_read, 
            df_bud_read, 
            'ORS Number', 
            'ORS_Ref', 
            'Amount', 
            'Gross_Amount', 
            cols_to_compare
        )
        process_end = time.time()
        
        print(f"✅ Reconciliation core logic took {process_end - process_start:.2f}s")
        print("-" * 30)
        print("Results Summary:")
        print(merged['Status'].value_counts())
        
        # Cleanup
        os.remove("stress_acc.xlsx")
        os.remove("stress_bud.xlsx")
        print("\n✅ Temporary test files cleaned up.")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    run_stress_test()
