import pandas as pd
import unittest
from reconciliation_core import reconcile_data

class TestDuplicateORS(unittest.TestCase):
    def setUp(self):
        # Common setup if needed, but we'll do per-test setup for clarity
        pass

    def test_case_a_perfect_duplicates(self):
        """
        Case A: Perfect Duplicates
        Accounting: 2 entries for ORS '100', both $50
        Budget:     2 entries for ORS '100', both $50
        Expected:   2 'Fully Matched' rows
        """
        print("\nRunning Test Case A: Perfect Duplicates")
        df_acc = pd.DataFrame({'ORS': ['100', '100'], 'Amount': [50.0, 50.0]})
        df_bud = pd.DataFrame({'ORS': ['100', '100'], 'Amount': [50.0, 50.0]})
        
        result, _ = reconcile_data(df_acc, df_bud, 'ORS', 'ORS', 'Amount', 'Amount', [])
        
        print(result[['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD', 'Status']])
        
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['Status'] == 'Fully Matched'))

    def test_case_b_mixed_amounts_reordering(self):
        """
        Case B: Mixed Amounts with Reordering
        Accounting: ORS '100' -> $50, $100
        Budget:     ORS '100' -> $100, $50 (Different order)
        Expected:   2 'Fully Matched' rows (Smart matching should align 50-50 and 100-100)
        """
        print("\nRunning Test Case B: Mixed Amounts (Reordering)")
        df_acc = pd.DataFrame({'ORS': ['100', '100'], 'Amount': [50.0, 100.0]})
        df_bud = pd.DataFrame({'ORS': ['100', '100'], 'Amount': [100.0, 50.0]})
        
        result, _ = reconcile_data(df_acc, df_bud, 'ORS', 'ORS', 'Amount', 'Amount', [])
        
        print(result[['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD', 'Status']])
        
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['Status'] == 'Fully Matched'))
        # Verify alignment
        row_50 = result[result['Clean_Amount_ACC'] == 50.0].iloc[0]
        self.assertEqual(row_50['Clean_Amount_BUD'], 50.0)

    def test_case_c_unmatched_counts(self):
        """
        Case C: Unmatched Counts (Orphans)
        Accounting: 3 entries for ORS '100'
        Budget:     2 entries for ORS '100'
        Expected:   2 Matches (or Mismatches), 1 'Missing in Budget'
        """
        print("\nRunning Test Case C: Unmatched Counts")
        df_acc = pd.DataFrame({'ORS': ['100', '100', '100'], 'Amount': [10.0, 20.0, 30.0]})
        df_bud = pd.DataFrame({'ORS': ['100', '100'], 'Amount': [10.0, 20.0]})
        
        result, _ = reconcile_data(df_acc, df_bud, 'ORS', 'ORS', 'Amount', 'Amount', [])
        
        print(result[['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD', 'Status']])
        
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[result['Status'] == 'Fully Matched']), 2)
        self.assertEqual(len(result[result['Status'] == 'Missing in Budget']), 1)
        # The missing one should be the $30 one
        missing_row = result[result['Status'] == 'Missing in Budget'].iloc[0]
        self.assertEqual(missing_row['Clean_Amount_ACC'], 30.0)

    def test_case_d_all_amount_mismatches(self):
        """
        Case D: All Amount Mismatches
        Accounting: ORS '100' -> $50, $50
        Budget:     ORS '100' -> $60, $70
        Expected:   2 'Amount Mismatch' rows
        """
        print("\nRunning Test Case D: All Amount Mismatches")
        df_acc = pd.DataFrame({'ORS': ['100', '100'], 'Amount': [50.0, 50.0]})
        df_bud = pd.DataFrame({'ORS': ['100', '100'], 'Amount': [60.0, 70.0]})
        
        result, _ = reconcile_data(df_acc, df_bud, 'ORS', 'ORS', 'Amount', 'Amount', [])
        
        print(result[['Clean_ORS', 'Clean_Amount_ACC', 'Clean_Amount_BUD', 'Status']])
        
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['Status'] == 'Amount Mismatch'))

if __name__ == '__main__':
    unittest.main()
