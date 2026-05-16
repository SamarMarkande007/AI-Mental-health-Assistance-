"""W1-DS2 | Hypothesis Testing Setup — Output: outputs/hypothesis_test_results.csv"""

import pandas as pd, numpy as np, os
from scipy.stats import chi2_contingency, ttest_ind

OUT = "outputs/w1_ds2_hypothesis_testing"
os.makedirs(OUT, exist_ok=True)

data = {
    "Age":            ["16-24","25-34","35-44","45-54","55-64","65-74","75+"],
    "Male_CMD_pct":   [10.0, 17.4, 16.3, 13.8, 15.6,  8.1,  5.6],
    "Female_CMD_pct": [28.2, 20.7, 22.3, 24.2, 20.2, 14.7, 11.0],
    "Male_N":         [249,  355,  468,  489,  541,  538,  418],
    "Female_N":       [311,  680,  712,  805,  685,  651,  644],
}
df = pd.DataFrame(data)
df["Male_CMD"]    = (df["Male_CMD_pct"]   / 100 * df["Male_N"]).round().astype(int)
df["Female_CMD"]  = (df["Female_CMD_pct"] / 100 * df["Female_N"]).round().astype(int)
df["Male_NoCMD"]  = df["Male_N"]   - df["Male_CMD"]
df["Female_NoCMD"]= df["Female_N"] - df["Female_CMD"]

# Test 1: Chi-square — Gender vs CMD
chi2_g, p_g, *_ = chi2_contingency([
    [df["Male_CMD"].sum(),   df["Male_NoCMD"].sum()],
    [df["Female_CMD"].sum(), df["Female_NoCMD"].sum()]
])

# Test 2: Chi-square — Age vs CMD
df["Total_CMD"]   = df["Male_CMD"]   + df["Female_CMD"]
df["Total_NoCMD"] = df["Male_NoCMD"] + df["Female_NoCMD"]
chi2_a, p_a, *_ = chi2_contingency(df[["Total_CMD","Total_NoCMD"]].values)

# Test 3: t-test — Male vs Female CMD rates
t_stat, p_t = ttest_ind(df["Male_CMD_pct"], df["Female_CMD_pct"])

def fmt_p(p): return "<0.001" if p < 0.001 else round(p, 4)
def verdict(p): return "Reject Ho (Significant)" if p < 0.05 else "Fail to Reject Ho"

results = pd.DataFrame({
    "Test":      ["Chi-square", "Chi-square", "t-test"],
    "Variables": ["Gender vs CMD", "Age vs CMD", "Male vs Female CMD rates"],
    "H0":        ["No gender-CMD association", "No age-CMD association", "Equal CMD rates by sex"],
    "Statistic": [round(chi2_g,4), round(chi2_a,4), round(t_stat,4)],
    "p_value":   [fmt_p(p_g), fmt_p(p_a), fmt_p(p_t)],
    "Conclusion":[verdict(p_g), verdict(p_a), verdict(p_t)],
})

out_path = f"{OUT}/hypothesis_test_results.csv"
results.to_csv(out_path, index=False)
print(f"[DS2] Done -> {out_path}")
print(results.to_string(index=False))
