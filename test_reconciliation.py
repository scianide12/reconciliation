import unittest
import pandas as pd
import numpy as np
from reconciliation_core import clean_currency, find_best_match, reconcile_data, clean_id

class TestReconciliationApp(unittest.TestCase):

    def test_clean_id(self):
        # Test ID cleaning logic (Text/Float artifacts)
        self.assertEqual(clean_id(123), "123")
        self.assertEqual(clean_id(123.0), "123")
        self.assertEqual(clean_id("123"), "123")
        self.assertEqual(clean_id("123.0"), "123")
        self.assertEqual(clean_id("  123  "), "123")
        self.assertEqual(clean_id(None), "")
        self.assertEqual(clean_id(np.nan), "")

    def test_clean_currency(self):
        # Test cleaning various currency formats
        self.assertEqual(clean_currency("$1,000.00"), 1000.0)
        self.assertEqual(clean_currency("1,234.56"), 1234.56)
        self.assertEqual(clean_currency("(500.00)"), -500.0)
        self.assertEqual(clean_currency("-100"), -100.0)
        self.assertEqual(clean_currency("invalid"), 0.0)
        self.assertEqual(clean_currency(None), 0.0)
        self.assertEqual(clean_currency(123.45), 123.45)

    def test_find_best_match(self):
        # Test column matching logic
        columns = ["ORS Number", "Amount", "Date", "Description", "ORS_Ref"]
        
        # Exact/Close match
        self.assertEqual(find_best_match(columns, ["ors", "ors no"]), 0) # Matches 'ORS Number' (index 0) because 'ors' in 'orsnumber'
        
        # Case insensitive
        self.assertEqual(find_best_match(columns, ["amount"]), 1)
        
        # Partial match
        self.assertEqual(find_best_match(columns, ["desc"]), 3)
        
        # Underscore handling
        self.assertEqual(find_best_match(columns, ["ors_ref"]), 4)
        
        # Spaced keywords
        self.assertEqual(find_best_match(columns, ["ors number"]), 0)
        
        # No match
        self.assertEqual(find_best_match(columns, ["xyz"]), 0) # Defaults to 0 if no match found

    def test_find_best_match_edge_cases(self):
        columns = ["Total Amount (USD)", "Trans. Date", "Ref #"]
        
        # Special characters handling
        self.assertEqual(find_best_match(columns, ["amount"]), 0) # 'amount' in 'totalamountusd'
        self.assertEqual(find_best_match(columns, ["date"]), 1)   # 'date' in 'transdate'
        self.assertEqual(find_best_match(columns, ["ref"]), 2)    # 'ref' in 'ref'


    def test_reconcile_data(self):
        # Create dummy dataframes
        data_acc = {
            'ORS': ['001', '002', '003', '005'],
            'Amount': [100.0, 200.0, 300.0, 500.0],
            'Payee': ['A', 'B', 'C', 'E']
        }
        data_bud = {
            'ORS No.': ['001', '002', '004', '005'],
            'Amount': [100.0, 250.0, 400.0, 500.0],
            'Payee': ['A', 'B', 'D', 'Different']
        }
        
        df_acc = pd.DataFrame(data_acc)
        df_bud = pd.DataFrame(data_bud)
        
        cols_to_compare = [{'acc_col': 'Payee', 'bud_col': 'Payee', 'display': 'Payee'}]
        
        merged, dropped = reconcile_data(
            df_acc, 
            df_bud, 
            'ORS', 
            'ORS No.', 
            'Amount', 
            'Amount', 
            cols_to_compare
        )
        
        # 1. Check Row Counts
        # Union of 001, 002, 003, 004, 005 = 5 rows
        self.assertEqual(len(merged), 5)
        
        # 2. Check Status
        # 001: Matched
        row_001 = merged[merged['Clean_ORS'] == '001'].iloc[0]
        self.assertEqual(row_001['Status'], 'Fully Matched')
        
        # 002: Amount Mismatch (200 vs 250)
        row_002 = merged[merged['Clean_ORS'] == '002'].iloc[0]
        self.assertEqual(row_002['Status'], 'Amount Mismatch')
        
        # 003: Missing in Budget
        row_003 = merged[merged['Clean_ORS'] == '003'].iloc[0]
        self.assertEqual(row_003['Status'], 'Missing in Budget')
        
        # 004: Missing in Accounting
        row_004 = merged[merged['Clean_ORS'] == '004'].iloc[0]
        self.assertEqual(row_004['Status'], 'Missing in Accounting')
        
        # 005: Data Mismatch (Payee: E vs Different)
        row_005 = merged[merged['Clean_ORS'] == '005'].iloc[0]
        self.assertEqual(row_005['Status'], 'Payee Mismatch')

    def test_reconcile_mixed_types(self):
        # Case: Accounting has MFO as int (123), Budget has MFO as float (123.0)
        df_acc = pd.DataFrame({'MFO': [123, 456], 'Amount': [100.0, 200.0]})
        df_bud = pd.DataFrame({'MFO': [123.0, 456.0], 'Amount': [100.0, 200.0]})
        
        result, _ = reconcile_data(df_acc, df_bud, 'MFO', 'MFO', 'Amount', 'Amount', [])
        
        # Both should match perfectly
        self.assertEqual(len(result), 2)
        self.assertTrue((result['Status'] == 'Fully Matched').all())

    def test_reconcile_data_dropped(self):
        # Create dummy dataframes with blanks
        data_acc = {
            'ORS': ['001', '002', ''], # 3rd is blank ORS
            'Amount': [100.0, None, 300.0] # 2nd is blank Amount
        }
        data_bud = {
            'ORS No.': ['001', '002', '003'],
            'Amount': [100.0, 200.0, 300.0]
        }
        
        df_acc = pd.DataFrame(data_acc)
        df_bud = pd.DataFrame(data_bud)
        
        merged, dropped = reconcile_data(
            df_acc, 
            df_bud, 
            'ORS', 
            'ORS No.', 
            'Amount', 
            'Amount', 
            []
        )
        
        # Check dropped
        # Acc row 2 (index 1) has blank Amount -> Drop
        # Acc row 3 (index 2) has blank ORS -> Drop
        self.assertEqual(len(dropped), 2)
        self.assertTrue('Source' in dropped.columns)
        self.assertEqual(dropped[dropped['Source'] == '(Acc)'].shape[0], 2)
        
        # Check merged
        # 001 should be processed (Matched)
        # 002 (Budget) should be Missing in Accounting (since Acc 002 dropped)
        # 003 (Budget) should be Missing in Accounting (since Acc 003 dropped)
        self.assertEqual(len(merged), 3)
        
        # Case: Text vs Number
        df_acc_2 = pd.DataFrame({'MFO': ["123", "456"], 'Amount': [100.0, 200.0]})
        df_bud_2 = pd.DataFrame({'MFO': [123.0, 456.0], 'Amount': [100.0, 200.0]})
        
        result_2, _ = reconcile_data(df_acc_2, df_bud_2, 'MFO', 'MFO', 'Amount', 'Amount', [])
        self.assertEqual(len(result_2), 2)
        self.assertTrue(all(result_2['Status'] == 'Fully Matched'))

    def test_reconcile_zero_dropped_refined(self):
        # Refined Logic:
        # Amount Column: Drop 0, 0.0, Blank
        # Other Columns: Drop Blank (Keep 0)
        
        data_acc = {
            'ORS': ['001', '002', '003', '004', '005'],
            'Amount': [
                0,           # Numeric zero -> Drop
                100.0,       # Valid -> Check other cols
                100.0,       # Valid
                100.0,       # Valid
                100.0        # Valid
            ],
            'Ref': [
                'A',         # Valid
                '',          # Empty -> Drop
                '0',         # "0" String -> Keep (User said only Amount 0 drops)
                0,           # 0 Numeric -> Keep (User said only Amount 0 drops)
                'B'          # Valid
            ]
        }
        
        # Budget side (simple)
        data_bud = {
            'ORS No.': ['005'],
            'Amount': [100.0]
        }
        
        df_acc = pd.DataFrame(data_acc)
        df_bud = pd.DataFrame(data_bud)
        
        # Mapping: ORS, Amount, Ref (as mapped col)
        cols_to_compare = [{'acc_col': 'Ref', 'bud_col': 'ORS No.', 'display': 'Ref'}] # Dummy mapping
        
        merged, dropped = reconcile_data(
            df_acc, 
            df_bud, 
            'ORS', 
            'ORS No.', 
            'Amount', 
            'Amount', 
            cols_to_compare
        )
        
        # Expected Drops:
        # 001: Amount is 0 -> Drop
        # 002: Ref is '' -> Drop
        # 003: Ref is '0' -> Keep
        # 004: Ref is 0 -> Keep
        # 005: Valid -> Keep
        
        # Dropped: 001, 002
        self.assertEqual(len(dropped), 2)
        dropped_ors = dropped[dropped['Source'] == '(Acc)']['ORS'].tolist()
        self.assertCountEqual(dropped_ors, ['001', '002'])
        
        # Verify 'Source' is the first column
        self.assertEqual(dropped.columns[0], 'Source')
        
        # Kept: 003, 004, 005
        # 005 matches Budget (Fully Matched)
        # 003, 004 (Missing in Budget)
        self.assertEqual(len(merged), 3) 
        merged_ors = merged['Clean_ORS'].tolist()
        self.assertCountEqual(merged_ors, ['003', '004', '005'])

    def test_reconcile_duplicates_swapped_order(self):
        # Scenario: ORS and Amount are identical (duplicates), but "Payee" differs.
        # We want to see if the engine pairs them intelligently or just by order.
        
        # Accounting: A -> Payee X, B -> Payee Y
        data_acc = {
            'ORS': ['001', '001'],
            'Amount': [100.0, 100.0],
            'Payee': ['Payee X', 'Payee Y']
        }
        
        # Budget: A -> Payee Y, B -> Payee X (Swapped Order)
        data_bud = {
            'ORS No.': ['001', '001'],
            'Amount': [100.0, 100.0],
            'Payee': ['Payee Y', 'Payee X']
        }
        
        df_acc = pd.DataFrame(data_acc)
        df_bud = pd.DataFrame(data_bud)
        
        cols_to_compare = [{'acc_col': 'Payee', 'bud_col': 'Payee', 'display': 'Payee'}]
        
        merged, _ = reconcile_data(
            df_acc, 
            df_bud, 
            'ORS', 
            'ORS No.', 
            'Amount', 
            'Amount', 
            cols_to_compare
        )
        
        # Current Logic Prediction:
        # It matches Acc[0] (Payee X) with Bud[0] (Payee Y) -> Mismatch
        # It matches Acc[1] (Payee Y) with Bud[1] (Payee X) -> Mismatch
        # Because it uses simple cumcount (0 with 0, 1 with 1)
        
        print("\n--- Duplicate Swap Test Results ---")
        # Note: Columns are suffixed _ACC and _BUD by reconcile_data
        print(merged[['Clean_ORS', 'Payee_ACC', 'Payee_BUD', 'Status']])
        
        # Smart Matching Check:
        # The engine should now sort by Payee before matching, so X matches X and Y matches Y.
        is_fully_matched = (merged['Status'] == 'Fully Matched').all()
        self.assertTrue(is_fully_matched, "Expected Smart Sorting to align duplicates correctly")

    def test_reconcile_combined_mismatch(self):
        # Specific test for combined mismatch
        df_acc = pd.DataFrame({
            'ORS': ['001'],
            'Amount': [100.0],
            'Payee': ['Payee A']
        })
        df_bud = pd.DataFrame({
            'ORS No.': ['001'],
            'Amount': [150.0], # Amount diff
            'Payee': ['Payee B'] # Data diff
        })
        
        cols_to_compare = [{'acc_col': 'Payee', 'bud_col': 'Payee', 'display': 'Payee'}]
        
        merged, _ = reconcile_data(df_acc, df_bud, 'ORS', 'ORS No.', 'Amount', 'Amount', cols_to_compare)
        
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]['Status'], 'Amount Mismatch, Payee Mismatch')


if __name__ == '__main__':
    unittest.main()
