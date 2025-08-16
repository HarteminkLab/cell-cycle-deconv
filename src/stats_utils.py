def test_z_proportion_change(subset_sample, total_sampled, 
    subset_population, total_population, print_results=True, alpha = 0.05):
    
    import numpy as np
    from scipy import stats
    from scipy.stats import chi2_contingency
    import math

    # Data
    x1, n1 = int(subset_sample), int(total_sampled) #68, 265 (68 successes out of 265 trials)
    x2, n2 = int(subset_population), int(total_population) # Sample 2: 448 successes out of 2353 trials

    ret_text = ""
    ret_text += (f"Sample 1: {x1}/{n1} = {x1/n1:.4f} ({x1/n1*100:.2f}%)\n")
    ret_text += (f"Sample 2: {x2}/{n2} = {x2/n2:.4f} ({x2/n2*100:.2f}%)\n")
    ret_text += ''

    # Method 1: Two-proportion z-test
    def two_prop_z_test(x1, n1, x2, n2):
        """
        Perform two-proportion z-test
        H0: p1 = p2 (proportions are equal)
        H1: p1 ≠ p2 (proportions are different)
        """
        p1 = x1 / n1
        p2 = x2 / n2

        # Pooled proportion
        p_pool = (x1 + x2) / (n1 + n2)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        # Z-statistic
        z = (p1 - p2) / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return z, p_value, p1, p2, p_pool

    # Perform two-proportion z-test
    z_stat, p_value_z, p1, p2, p_pool = two_prop_z_test(x1, n1, x2, n2)

    ret_text += ("Method 1: Two-Proportion Z-Test\n")
    ret_text += (f"Z-statistic: {z_stat:.4f}\n")
    ret_text += (f"P-value: {p_value_z:.6f}\n")
    ret_text += (f"Pooled proportion: {p_pool:.4f}\n")
    ret_text += ("\n")

    # Method 2: Chi-square test of independence
    # Create contingency table
    contingency_table = np.array([
        [x1, n1 - x1],      # Sample 1: [successes, failures]
        [x2, n2 - x2]       # Sample 2: [successes, failures]
    ])

    ret_text += ("Method 2: Chi-Square Test of Independence\n")
    ret_text += ("Contingency table:\n")
    ret_text += ("               Success  Failure\n")
    ret_text += (f"Sample 1:      {x1:7d}  {n1-x1:7d}\n")
    ret_text += (f"Sample 2:      {x2:7d}  {n2-x2:7d}\n")
    ret_text += ("\n")

    chi2_stat, p_value_chi2, dof, expected = chi2_contingency(contingency_table)

    ret_text += (f"Chi-square statistic: {chi2_stat:.4f}\n")
    ret_text += (f"P-value: {p_value_chi2:.6f}\n")
    ret_text += (f"Degrees of freedom: {dof}\n")
    ret_text += ("\n")

    # Method 3: Fisher's exact test (for comparison, though not necessary with large samples)
    from scipy.stats import fisher_exact

    # Fisher's exact test expects a 2x2 contingency table
    fisher_stat, p_value_fisher = fisher_exact(contingency_table)

    ret_text += ("Method 3: Fisher's Exact Test (for comparison)\n")
    ret_text += (f"Odds ratio: {fisher_stat:.4f}\n")
    ret_text += (f"P-value: {p_value_fisher:.6f}\n")
    ret_text += ("\n")

    # Interpretation
    ret_text += (f"Interpretation (α = {alpha}):\n")
    ret_text += ("-" * 40)

    if p_value_fisher < alpha:
        reject_null = True
        ret_text += ("✓ REJECT the null hypothesis\n")
        ret_text += ("  The proportions are significantly different\n")
        ret_text += (f"  (p = {p_value_fisher:.6f} < {alpha})\n")
    else:
        reject_null = False
        ret_text += ("✓ REJECT the null hypothesis\n")
        ret_text += ("✗ FAIL TO REJECT the null hypothesis\n")
        ret_text += ("  No significant difference between proportions\n")
        ret_text += (f"  (p = {p_value_fisher:.6f} ≥ {alpha})\n")

    ret_text += ("\n")
    ret_text += ("Effect size:\n")
    difference = abs(p1 - p2)
    ret_text += (f"Absolute difference: {difference:.4f} ({difference*100:.2f} percentage points)\n")

    # Cohen's h for effect size
    cohens_h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    ret_text += (f"Cohen's h: {cohens_h:.4f}\n")

    if abs(cohens_h) < 0.2:
        effect_size = "small"
    elif abs(cohens_h) < 0.5:
        effect_size = "small to medium"
    elif abs(cohens_h) < 0.8:
        effect_size = "medium to large"
    else:
        effect_size = "large"

    ret_text += (f"Effect size interpretation: {effect_size}\n")

    if print_results:
        print(ret_text)

    return ret_text, reject_null, p_value_fisher

