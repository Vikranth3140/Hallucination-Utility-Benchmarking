"""Compute Cohen's kappa and general agreement metrics."""
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator1", type=str, required=True, help="Annotator 1 CSV")
    parser.add_argument("--annotator2", type=str, help="Optionally Annotator 2 CSV for human-human")
    return parser.parse_args()

def main():
    args = parse_args()
    
    df1 = pd.read_csv(args.annotator1)
    
    if args.annotator2:
        df2 = pd.read_csv(args.annotator2)
        # Merge on sample_id
        df = pd.merge(df1, df2, on="sample_id", suffixes=('_a1', '_a2'))
        
        # Agreement between humans
        u_true1 = df['annotator_unsupported_label_a1'].fillna(-1).astype(int)
        u_true2 = df['annotator_unsupported_label_a2'].fillna(-1).astype(int)
        
        # for utility_label, only when unsupported_label = 1
        mask_u1 = (u_true1 == 1) & (u_true2 == 1)
        ut1 = df.loc[mask_u1, 'annotator_utility_label_a1'].fillna('None').astype(str)
        ut2 = df.loc[mask_u1, 'annotator_utility_label_a2'].fillna('None').astype(str)
        
        print("=== Human vs Human Agreement ===")
        print(f"Raw Agreement (Unsupported): {accuracy_score(u_true1, u_true2):.4f}")
        def safe_kappa(y1, y2):
            try:
                if len(set(y1).union(set(y2))) <= 1:
                    return 1.0 # Perfect agreement if all are the same and match
            except:
                pass
            return cohen_kappa_score(y1, y2)

        print(f"Cohen's Kappa (Unsupported): {safe_kappa(u_true1, u_true2):.4f}")
        
        if mask_u1.sum() > 0:
            print(f"Raw Agreement (Utility): {accuracy_score(ut1, ut2):.4f}")
            print(f"Cohen's Kappa (Utility): {safe_kappa(ut1, ut2):.4f}")
        else:
            print("No overlapping unsupported=1 examples for utility kappa.")
            
    else:
        # Judge vs Human
        u_true = df1['annotator_unsupported_label'].fillna(-1).astype(int)
        u_pred = df1['existing_judge_unsupported_label'].fillna(-1).astype(int)
        
        mask_u1 = (u_true == 1) & (u_pred == 1)
        ut_true = df1.loc[mask_u1, 'annotator_utility_label'].fillna('None').astype(str)
        ut_pred = df1.loc[mask_u1, 'existing_judge_utility_label'].fillna('None').astype(str)
        
        print("=== Judge vs Human Agreement ===")
        print(f"Raw Agreement (Unsupported): {accuracy_score(u_true, u_pred):.4f}")
        
        def safe_kappa(y1, y2):
            try:
                if len(set(y1).union(set(y2))) <= 1:
                    return 1.0
            except:
                pass
            return cohen_kappa_score(y1, y2)
            
        print(f"Cohen's Kappa (Unsupported): {safe_kappa(u_true, u_pred):.4f}")
        
        if mask_u1.sum() > 0:
            print(f"Raw Agreement (Utility): {accuracy_score(ut_true, ut_pred):.4f}")
            print(f"Cohen's Kappa (Utility): {safe_kappa(ut_true, ut_pred):.4f}")

if __name__ == "__main__":
    main()
