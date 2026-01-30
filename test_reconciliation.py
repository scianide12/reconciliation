import unittest
import pandas as pd
import numpy as np
from reconciliation_core import clean_currency, find_best_match, reconcile_data

class TestReconciliationApp(unittest.TestCase):

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
        
        merged = reconcile_data(
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
        
        merged = reconcile_data(df_acc, df_bud, 'ORS', 'ORS No.', 'Amount', 'Amount', cols_to_compare)
        
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]['Status'], 'Amount Mismatch, Payee Mismatch')


if __name__ == '__main__':
    unittest.main()
