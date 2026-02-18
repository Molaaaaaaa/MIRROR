"""
MIRROR Results Analyzer
Analyze prediction accuracy by category, method, and other dimensions.
"""
import os
import argparse
import pandas as pd
from typing import Dict
from tabulate import tabulate


def load_all_predictions(results_dir: str, method: str = None, setting: str = None) -> pd.DataFrame:
    """
    Load all prediction CSV files from results directory.
    
    Args:
        results_dir: Path to results directory
        method: Filter by method (e.g., 'MIRROR', 'RER')
        setting: Filter by setting (e.g., 'S1', 'NonTarget')
    """
    all_dfs = []
    
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # Parse method and setting from directory name
        parts = subdir.split('_')
        dir_method = parts[0]
        dir_setting = '_'.join(parts[1:]) if len(parts) > 1 else 'S1'
        
        # Apply filters
        if method and dir_method != method:
            continue
        if setting and dir_setting != setting:
            continue
        
        # Load all CSV files in this directory
        for fname in os.listdir(subdir_path):
            if fname.startswith('pred_') and fname.endswith('.csv'):
                fpath = os.path.join(subdir_path, fname)
                try:
                    df = pd.read_csv(fpath, encoding='utf-8-sig')
                    df['Dir_Method'] = dir_method
                    df['Dir_Setting'] = dir_setting
                    all_dfs.append(df)
                except Exception as e:
                    print(f"[Warning] Failed to load {fpath}: {e}")
    
    if not all_dfs:
        print("[Error] No prediction files found")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined)} predictions from {len(all_dfs)} files")
    return combined


def analyze_by_category(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """Analyze accuracy by category."""
    if df.empty:
        return pd.DataFrame()
    
    # Filter valid predictions
    valid_df = df[df['Ground_Truth'] != 'N/A'].copy()
    
    results = []
    for category in sorted(valid_df['Category'].unique()):
        cat_df = valid_df[valid_df['Category'] == category]
        
        total = len(cat_df)
        if total < min_samples:
            continue
        
        correct = cat_df['Is_Correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Count unique questions and students
        n_questions = cat_df['Question'].nunique()
        n_students = cat_df['StudentID'].nunique()
        
        results.append({
            'Category': category,
            'Accuracy': round(accuracy, 2),
            'Correct': int(correct),
            'Total': total,
            'Questions': n_questions,
            'Students': n_students
        })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('Accuracy', ascending=False)
    
    return result_df


def analyze_by_category_and_method(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze accuracy by category and method."""
    if df.empty:
        return pd.DataFrame()
    
    valid_df = df[df['Ground_Truth'] != 'N/A'].copy()
    
    results = []
    for category in sorted(valid_df['Category'].unique()):
        for method in sorted(valid_df['Dir_Method'].unique()):
            subset = valid_df[(valid_df['Category'] == category) & (valid_df['Dir_Method'] == method)]
            
            total = len(subset)
            if total == 0:
                continue
            
            correct = subset['Is_Correct'].sum()
            accuracy = (correct / total * 100) if total > 0 else 0
            
            results.append({
                'Category': category,
                'Method': method,
                'Accuracy': round(accuracy, 2),
                'Correct': int(correct),
                'Total': total
            })
    
    return pd.DataFrame(results)


def analyze_by_question(df: pd.DataFrame, top_n: int = 20) -> Dict[str, pd.DataFrame]:
    """Analyze accuracy by individual question - best and worst performing."""
    if df.empty:
        return {}
    
    valid_df = df[df['Ground_Truth'] != 'N/A'].copy()
    
    results = []
    for question in valid_df['Question'].unique():
        q_df = valid_df[valid_df['Question'] == question]
        
        total = len(q_df)
        if total < 5:  # Need at least 5 samples
            continue
        
        correct = q_df['Is_Correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        category = q_df['Category'].iloc[0]
        
        # Truncate question for display
        q_short = question[:80] + '...' if len(question) > 80 else question
        
        results.append({
            'Question': q_short,
            'Category': category,
            'Accuracy': round(accuracy, 2),
            'Correct': int(correct),
            'Total': total
        })
    
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return {}
    
    result_df = result_df.sort_values('Accuracy', ascending=False)
    
    return {
        'best': result_df.head(top_n),
        'worst': result_df.tail(top_n).sort_values('Accuracy')
    }


def analyze_by_student(df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
    """Analyze accuracy by student - best and worst performing."""
    if df.empty:
        return {}
    
    valid_df = df[df['Ground_Truth'] != 'N/A'].copy()
    
    results = []
    for student_id in valid_df['StudentID'].unique():
        s_df = valid_df[valid_df['StudentID'] == student_id]
        
        total = len(s_df)
        correct = s_df['Is_Correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        results.append({
            'StudentID': student_id,
            'Accuracy': round(accuracy, 2),
            'Correct': int(correct),
            'Total': total
        })
    
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return {}
    
    result_df = result_df.sort_values('Accuracy', ascending=False)
    
    return {
        'best': result_df.head(top_n),
        'worst': result_df.tail(top_n).sort_values('Accuracy')
    }


def compare_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Compare accuracy across different methods."""
    if df.empty:
        return pd.DataFrame()
    
    valid_df = df[df['Ground_Truth'] != 'N/A'].copy()
    
    results = []
    for method in sorted(valid_df['Dir_Method'].unique()):
        m_df = valid_df[valid_df['Dir_Method'] == method]
        
        total = len(m_df)
        correct = m_df['Is_Correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        n_students = m_df['StudentID'].nunique()
        n_categories = m_df['Category'].nunique()
        
        results.append({
            'Method': method,
            'Accuracy': round(accuracy, 2),
            'Correct': int(correct),
            'Total': total,
            'Students': n_students,
            'Categories': n_categories
        })
    
    return pd.DataFrame(results).sort_values('Accuracy', ascending=False)


def generate_category_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pivot table: Category x Method."""
    if df.empty:
        return pd.DataFrame()
    
    cat_method_df = analyze_by_category_and_method(df)
    if cat_method_df.empty:
        return pd.DataFrame()
    
    pivot = cat_method_df.pivot(index='Category', columns='Method', values='Accuracy')
    return pivot


def print_summary(df: pd.DataFrame, title: str = "Analysis Results"):
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    if df.empty:
        print("No data available")
        return
    
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


def main():
    parser = argparse.ArgumentParser(description="MIRROR Results Analyzer")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Path to results directory")
    parser.add_argument("--method", type=str, default=None,
                        help="Filter by method (e.g., MIRROR, RER)")
    parser.add_argument("--setting", type=str, default=None,
                        help="Filter by setting (e.g., S1, NonTarget)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save analysis to CSV file")
    parser.add_argument("--by-category", action="store_true", default=True,
                        help="Analyze by category")
    parser.add_argument("--by-question", action="store_true",
                        help="Analyze by individual question")
    parser.add_argument("--by-student", action="store_true",
                        help="Analyze by student")
    parser.add_argument("--compare-methods", action="store_true",
                        help="Compare different methods")
    parser.add_argument("--pivot", action="store_true",
                        help="Generate Category x Method pivot table")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top/bottom items to show")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from: {args.results_dir}")
    df = load_all_predictions(args.results_dir, method=args.method, setting=args.setting)
    
    if df.empty:
        print("[Error] No data loaded")
        return
    
    # Print basic stats
    valid_df = df[df['Ground_Truth'] != 'N/A']
    total = len(valid_df)
    correct = valid_df['Is_Correct'].sum()
    overall_acc = (correct / total * 100) if total > 0 else 0
    
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total Predictions: {total}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Students: {valid_df['StudentID'].nunique()}")
    print(f"Categories: {valid_df['Category'].nunique()}")
    print(f"Questions: {valid_df['Question'].nunique()}")
    
    # Run requested analyses
    all_results = {}
    
    if args.by_category or args.all:
        cat_df = analyze_by_category(df)
        print_summary(cat_df, "ACCURACY BY CATEGORY")
        all_results['by_category'] = cat_df
    
    if args.compare_methods or args.all:
        method_df = compare_methods(df)
        print_summary(method_df, "ACCURACY BY METHOD")
        all_results['by_method'] = method_df
    
    if args.pivot or args.all:
        pivot_df = generate_category_pivot(df)
        if not pivot_df.empty:
            print("\n" + "=" * 80)
            print("CATEGORY x METHOD PIVOT TABLE")
            print("=" * 80)
            print(tabulate(pivot_df, headers='keys', tablefmt='grid', showindex=True))
            all_results['pivot'] = pivot_df
    
    if args.by_question or args.all:
        q_results = analyze_by_question(df, top_n=args.top_n)
        if q_results:
            print_summary(q_results['best'], f"TOP {args.top_n} BEST PERFORMING QUESTIONS")
            print_summary(q_results['worst'], f"TOP {args.top_n} WORST PERFORMING QUESTIONS")
            all_results['questions_best'] = q_results['best']
            all_results['questions_worst'] = q_results['worst']
    
    if args.by_student or args.all:
        s_results = analyze_by_student(df, top_n=args.top_n)
        if s_results:
            print_summary(s_results['best'], f"TOP {args.top_n} BEST PERFORMING STUDENTS")
            print_summary(s_results['worst'], f"TOP {args.top_n} WORST PERFORMING STUDENTS")
            all_results['students_best'] = s_results['best']
            all_results['students_worst'] = s_results['worst']
    
    # Save results
    if args.output:
        output_dir = os.path.dirname(args.output) or '.'
        base_name = os.path.splitext(os.path.basename(args.output))[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, result_df in all_results.items():
            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                out_path = os.path.join(output_dir, f"{base_name}_{name}.csv")
                result_df.to_csv(out_path, index=True, encoding='utf-8-sig')
                print(f"Saved: {out_path}")
        
        print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
