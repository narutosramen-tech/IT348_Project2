#!/usr/bin/env python3
"""
Example usage of drift-based retraining system with real malware data.
Shows how to implement sustainable ML for malware detection.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_all_years_data, print_data_summary
from retraining_system import DriftAwareRetrainingSystem

def main():
    print("="*80)
    print("DRIFT-BASED RETRAINING FOR MALWARE DETECTION")
    print("Sustainable ML Approach with Real Data")
    print("="*80)

    # Step 1: Load the real data
    print("\n1. LOADING MALWARE DETECTION DATA")
    print("-"*50)

    data_folder = "input_data"
    print(f"Loading data from: {data_folder}")

    try:
        all_data = get_all_years_data(data_folder)
        print_data_summary(all_data)

        years = sorted(list(all_data.keys()))
        print(f"\nAvailable years: {years}")
        print(f"Total years: {len(years)}")

        # Show class distribution
        print("\nClass distribution across years:")
        for year in years:
            X, y = all_data[year]
            benign_count = (y == 0).sum()
            malware_count = (y == 1).sum()
            total = benign_count + malware_count
            print(f"  {year}: {total} total, {malware_count} malware ({malware_count/total:.1%})")

    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Using synthetic data for demonstration...")
        # Create synthetic data for demonstration
        all_data = {}
        for year in ["2014", "2015", "2016", "2017", "2018", "2019", "2020"]:
            n_samples = np.random.randint(200, 1000)
            n_features = 100
            X = pd.DataFrame(np.random.randn(n_samples, n_features),
                           columns=[f"feature_{i:03d}" for i in range(n_features)])
            # Simulate imbalanced malware data (~10% malware)
            y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]))
            all_data[year] = (X, y)
        years = sorted(list(all_data.keys()))
        print(f"Created synthetic data for years: {years}")

    # Step 2: Initialize the retraining system
    print("\n\n2. INITIALIZING RETRAINING SYSTEM")
    print("-"*50)

    # Configuration parameters
    drift_threshold = 0.3  # 30% feature drift triggers retraining
    performance_threshold = 0.05  # 5% performance drop triggers retraining
    registry_path = "malware_detection_registry"

    print(f"Configuration:")
    print(f"  • Drift threshold: {drift_threshold:.0%} (retrain if >{drift_threshold*100:.0f}% features drift)")
    print(f"  • Performance threshold: {performance_threshold:.0%} (retrain if performance drops >{performance_threshold*100:.0f}%)")
    print(f"  • Model registry: {registry_path}")
    print(f"  • Model types to train: LogisticRegression, RandomForest")

    retraining_system = DriftAwareRetrainingSystem(
        drift_threshold=drift_threshold,
        performance_degradation_threshold=performance_threshold,
        registry_path=registry_path
    )

    # Step 3: Run progressive validation
    print("\n\n3. RUNNING PROGRESSIVE VALIDATION PIPELINE")
    print("-"*50)
    print("Strategy: Train on historical data, validate on current year")
    print("Start from 2016 (needs at least 2 years of training data)")

    # Run the pipeline
    pipeline_results = retraining_system.run_progressive_validation_pipeline(
        all_data=all_data,
        start_year="2016",  # Skip first few years for sufficient training history
        end_year="2020",
        model_types=["LogisticRegression", "RandomForest"]
    )

    # Step 4: Analyze results
    print("\n\n4. ANALYSIS OF RETRAINING DECISIONS")
    print("-"*50)

    summary_df = pipeline_results.get("summary", pd.DataFrame())
    if not summary_df.empty:
        print("\nYear-by-year retraining decisions:")
        for _, row in summary_df.iterrows():
            status_symbol = "🔄" if row["status"] == "retrained" else "⏸️"
            print(f"  {row['year']}: {status_symbol} {row['status']} - {row['reason']}")

        # Calculate statistics
        total_years = len(summary_df)
        retrained_years = len(summary_df[summary_df["status"] == "retrained"])
        skipped_years = len(summary_df[summary_df["status"] == "skipped"])

        print(f"\nSummary statistics:")
        print(f"  • Total years processed: {total_years}")
        print(f"  • Years retrained: {retrained_years} ({retrained_years/total_years:.0%})")
        print(f"  • Years skipped: {skipped_years} ({skipped_years/total_years:.0%})")

        if "drift_rate" in summary_df.columns:
            avg_drift = summary_df["drift_rate"].mean()
            print(f"  • Average drift rate: {avg_drift:.1%}")

    # Step 5: Model Registry Analysis
    print("\n\n5. MODEL REGISTRY ANALYSIS")
    print("-"*50)

    registry_df = retraining_system.registry.list_models()
    if not registry_df.empty:
        print(f"\nTotal registered models: {len(registry_df)}")

        # Show by model type
        model_counts = registry_df["model_name"].value_counts()
        print("\nModels by type:")
        for model_name, count in model_counts.items():
            print(f"  • {model_name}: {count} versions")

        # Show active models
        active_models = registry_df[registry_df["is_active"] == True]
        if not active_models.empty:
            print("\nCurrent active models:")
            for _, model in active_models.iterrows():
                print(f"  • {model['model_name']} v{model['version']} (trained on {len(model['training_years'])} years)")

        # Show performance trends
        print("\nPerformance trends (F1-score):")
        for model_name in ["LogisticRegression", "RandomForest"]:
            history_df = retraining_system.registry.get_model_performance_history(model_name)
            if not history_df.empty and "f1_score" in history_df.columns:
                best_f1 = history_df["f1_score"].max()
                latest_f1 = history_df.iloc[-1]["f1_score"] if len(history_df) > 0 else 0
                print(f"  • {model_name}: Best={best_f1:.3f}, Latest={latest_f1:.3f}")

    # Step 6: Implementation Recommendations
    print("\n\n6. IMPLEMENTATION RECOMMENDATIONS")
    print("-"*50)

    print("\nFor production deployment, consider:")

    print("\nA. Monitoring Setup:")
    print("   1. Continuous drift monitoring")
    print("   2. Real-time performance tracking")
    print("   3. Alert system for critical degradation")

    print("\nB. Deployment Strategy:")
    print("   1. Canary deployments (5% → 25% → 100%)")
    print("   2. A/B testing for new models")
    print("   3. Rollback capabilities")

    print("\nC. Operational Considerations:")
    print("   1. Retraining frequency: Quarterly review")
    print("   2. Performance targets: F1 > 0.85")
    print("   3. Maximum allowable false positive rate: < 1%")

    print("\nD. Data Management:")
    print("   1. Retain labeled malware samples for future training")
    print("   2. Regularly update feature set based on new API calls")
    print("   3. Monitor data quality and completeness")

    # Step 7: Next Steps
    print("\n\n7. NEXT STEPS FOR YOUR IMPLEMENTATION")
    print("-"*50)

    print("""
    1. Test with full dataset: Run the pipeline on all years (2014-2021)
    2. Fine-tune thresholds: Adjust based on your risk tolerance
    3. Add more model types: Include SVM, Gradient Boosting, Neural Networks
    4. Implement ensemble methods: Combine multiple models for better robustness
    5. Add explainability: Integrate SHAP/LIME for model interpretability
    6. Set up automation: Schedule periodic drift checks
    7. Create monitoring dashboard: Track key metrics over time
    """)

    print("\n" + "="*80)
    print("DRIFT-BASED RETRAINING IMPLEMENTATION COMPLETE")
    print("="*80)
    print("""
    Key benefits of this approach:
    1. Adaptive: Retrains only when needed (saves resources)
    2. Sustainable: Maintains performance over time
    3. Explainable: Clear reasons for retraining decisions
    4. Scalable: Handles multi-year data automatically
    5. Robust: Combines drift detection with performance monitoring
    """)


def quick_start_guide():
    """Quick start guide for using the retraining system."""
    print("\n" + "="*80)
    print("QUICK START GUIDE")
    print("="*80)

    print("""
    1. Basic Usage:
        ```
        from data import get_all_years_data
        from retraining_system import DriftAwareRetrainingSystem

        # Load data
        all_data = get_all_years_data("input_data")

        # Initialize system
        retrainer = DriftAwareRetrainingSystem()

        # Run for a specific year
        result = retrainer.retrain_with_drift_awareness(
            all_data=all_data,
            current_year="2020"
        )
        ```

    2. Run Progressive Validation:
        ```
        # Run across multiple years
        results = retrainer.run_progressive_validation_pipeline(
            all_data=all_data,
            start_year="2016",
            end_year="2020"
        )
        ```

    3. Check Model Registry:
        ```
        # List all registered models
        models_df = retrainer.registry.list_models()
        print(models_df)

        # Get performance history
        history_df = retrainer.registry.get_model_performance_history("RandomForest")
        print(history_df)
        ```

    4. Manual Retraining Trigger:
        ```
        # Force retraining
        result = retrainer.retrain_with_drift_awareness(
            all_data=all_data,
            current_year="2021",
            force_retrain=True
        )
        ```
    """)


if __name__ == "__main__":
    main()
    quick_start_guide()