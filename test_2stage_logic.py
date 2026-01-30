import pandas as pd
from reconciliation_core import reconcile_data

# --- TEST SCENARIO: 2-Stage Matching ---
# Goal: Verify that exact amounts are prioritized, and remaining duplicates are matched by order.

# 1. Setup Data
# Accounting: Two entries for ORS #100. One is $50, One is $99.
data_acc = {
    'ORS': ['100', '100', '200'],
    'Amount': [50.00, 99.00, 500.00],
    'Details': ['Match', 'Mismatch', 'Acc Orphan']
}

# Budget: Two entries for ORS #100. One is $50 (Match), One is $88 (Mismatch).
# Note: We put the $88 FIRST to prove that order doesn't break the exact match logic.
data_bud = {
    'ORS': ['100', '100', '300'],
    'Amount': [88.00, 50.00, 300.00],
    'Details': ['Mismatch', 'Match', 'Bud Orphan']
}

df_acc = pd.DataFrame(data_acc)
df_bud = pd.DataFrame(data_bud)

print("--- INPUT DATA ---")
print("Accounting:")
print(df_acc)
print("\nBudget:")
print(df_bud)
print("-" * 30)

# 2. Run Reconciliation
# We map 'Details' to 'Details' to check row integrity
cols_to_compare = [{'acc_col': 'Details', 'bud_col': 'Details', 'norm': 'details'}]

result, _ = reconcile_data(
    df_acc, 
    df_bud, 
    acc_ors_col='ORS', 
    bud_ors_col='ORS', 
    acc_amt_col='Amount', 
    bud_amt_col='Amount', 
    cols_to_compare=cols_to_compare
)

# 3. Analyze Results
print("\n--- RESULTS ---")
columns_to_show = ['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD', 'Details_ACC', 'Details_BUD', 'Status']
print(result[columns_to_show])

print("\n--- VERIFICATION ---")
# Check 1: The $50 pair should be "Matched"
match_row = result[(result['Clean_ORS'] == '100') & (result['Clean_Amount_ACC'] == 50.0)]
if not match_row.empty and match_row.iloc[0]['Status'] == 'Fully Matched':
    print("✅ CHECK 1 PASS: Perfect match found (ORS 100, $50).")
else:
    print("❌ CHECK 1 FAIL: Perfect match not handled correctly.")

# Check 2: The Mismatch pair ($99 vs $88) should be paired but have "Amount Mismatch" status
mismatch_row = result[(result['Clean_ORS'] == '100') & (result['Clean_Amount_ACC'] == 99.0)]
if not mismatch_row.empty and 'Amount Mismatch' in mismatch_row.iloc[0]['Status']:
    print("✅ CHECK 2 PASS: Fallback match found (ORS 100, $99 vs $88).")
else:
    print("❌ CHECK 2 FAIL: Fallback matching failed.")

# Check 3: Orphans
if len(result[result['Status'] == 'Missing in Budget']) == 1:
    print("✅ CHECK 3 PASS: Correct orphan count.")
else:
    print("❌ CHECK 3 FAIL: Orphan count wrong.")
