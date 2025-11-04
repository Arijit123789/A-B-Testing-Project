import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

st.title("A/B Testing Simulation App")

# Input parameters
n_A = st.number_input("Visitors in Variant A", value=10000)
n_B = st.number_input("Visitors in Variant B", value=10000)
p_A_true = st.number_input("True Conversion Rate of Variant A", value=0.10)
p_B_true = st.number_input("True Conversion Rate of Variant B", value=0.12)

# Run Simulation
if st.button("Run A/B Test Simulation"):
    np.random.seed(42)
    conversions_A = np.random.binomial(n_A, p_A_true)
    conversions_B = np.random.binomial(n_B, p_B_true)

    CR_A = conversions_A / n_A
    CR_B = conversions_B / n_B

    # Confidence Intervals
    ci_A_low, ci_A_upp = proportion_confint(conversions_A, n_A, alpha=0.05)
    ci_B_low, ci_B_upp = proportion_confint(conversions_B, n_B, alpha=0.05)

    # Z-Test
    count = np.array([conversions_A, conversions_B])
    nobs = np.array([n_A, n_B])
    stat, pval = proportions_ztest(count, nobs)

    st.subheader("Results Summary")
    st.write(f"Variant A Conversion Rate: **{CR_A:.2%}**  (95% CI: {ci_A_low:.2%} – {ci_A_upp:.2%})")
    st.write(f"Variant B Conversion Rate: **{CR_B:.2%}**  (95% CI: {ci_B_low:.2%} – {ci_B_upp:.2%})")
    st.write(f"Z-Statistic: **{stat:.4f}**,  P-Value: **{pval:.6f}**")

    # Decision
    if pval < 0.05:
        st.success("✅ Reject the Null Hypothesis — Variant B performs significantly better!")
    else:
        st.warning("❌ Fail to Reject Null Hypothesis — Difference not statistically significant.")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(['Variant A', 'Variant B'], [CR_A, CR_B], yerr=[
        [CR_A - ci_A_low, CR_B - ci_B_low],
        [ci_A_upp - CR_A, ci_B_upp - CR_B]
    ], capsize=10, color=['skyblue', 'lightcoral'])
    ax.set_ylabel("Conversion Rate")
    st.pyplot(fig)
