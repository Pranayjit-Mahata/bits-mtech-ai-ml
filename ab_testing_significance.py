"""
Module: ab_testing_significance.py
Course Reference: Introduction to Statistical Methods (ISM)
Description:
    A production-grade implementation of a two-proportion Z-test used 
    in A/B testing evaluation. This framework determines if the conversion 
    rate variant (Group B) yields a statistically significant improvement 
    over the baseline control (Group A).

"""

import numpy as np
import scipy.stats as stats


def evaluate_ab_test(
    control_users: int,
    control_conversions: int,
    variant_users: int,
    variant_conversions: int,
    alpha: float = 0.05,
) -> dict:
    """
    Computes the pooled conversion rate, Z-statistic, and p-value for a two-tailed test.

    Parameters:
        control_users (int): Total sample count for the control group (A).
        control_conversions (int): Successful conversions in control group.
        variant_users (int): Total sample count for the variant group (B).
        variant_conversions (int): Successful conversions in variant group.
        alpha (float): Significance level boundary (default 5%).

    Returns:
        dict: Detailed statistical metrics from the experiment.
    """
    # 1. Calculate individual conversion probabilities (p_hat)
    p_control = control_conversions / control_users
    p_variant = variant_conversions / variant_users

    # 2. Calculate the pooled conversion rate (p_pooled) under Null Hypothesis H0
    p_pooled = (control_conversions + variant_conversions) / (
        control_users + variant_users
    )

    # 3. Standard error computation
    standard_error = np.sqrt(
        p_pooled * (1 - p_pooled) * (1 / control_users + 1 / variant_users)
    )

    # 4. Compute Z-Score statistic
    z_score = (p_variant - p_control) / standard_error

    # 5. Compute Two-Tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # 6. Evaluate statistical significance boundary
    is_significant = p_value < alpha

    return {
        "Control Conversion Rate": round(p_control, 4),
        "Variant Conversion Rate": round(p_variant, 4),
        "Z-Statistic": round(z_score, 4),
        "p-value": round(p_value, 6),
        "Statistically Significant": is_significant,
    }


if __name__ == "__main__":
    # Mock Scenario: Evaluating a new recommendation algorithm deployment
    # Control Group (A): Old Algorithm | Variant Group (B): New AI Model
    CONTROL_TOTAL, CONTROL_SUCCESS = 10000, 1200  # 12% Conversion Rate
    VARIANT_TOTAL, VARIANT_SUCCESS = 10500, 1386  # 13.2% Conversion Rate

    results = evaluate_ab_test(
        control_users=CONTROL_TOTAL,
        control_conversions=CONTROL_SUCCESS,
        variant_users=VARIANT_TOTAL,
        variant_conversions=VARIANT_SUCCESS,
        alpha=0.05,
    )

    print("\n--- A/B Test Statistical Evaluation Results ---")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    print("\nConclusion:")
    if results["Statistically Significant"]:
        print(
            "Reject the Null Hypothesis (H0). The change yields a genuine performance uplift."
        )
    else:
        print(
            "Fail to reject the Null Hypothesis (H0). The change might be due to random noise."
        )
