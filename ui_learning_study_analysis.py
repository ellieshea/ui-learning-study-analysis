# SETUP — install/import libraries
!pip install scipy statsmodels pandas matplotlib seaborn --quiet

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

# LOADING DATA
files_list = os.listdir()
minimal_file = [f for f in files_list if f.startswith("Minimal")][0]
scaffolded_file = [f for f in files_list if f.startswith("Scaffolded")][0]
gamified_file = [f for f in files_list if f.startswith("Gamified")][0]

df_minimal = pd.read_csv(minimal_file)
df_scaffolded = pd.read_csv(scaffolded_file)
df_gamified = pd.read_csv(gamified_file)

# NOTE: Gamified's Score column reflects a differently-themed puzzle
# (dog/toy/breed matching instead of fruit/hat matching) -- this is by
# design (the gamified condition uses game-like theming), not an error.

df_minimal["condition"] = "minimal"
df_scaffolded["condition"] = "scaffolded"
df_gamified["condition"] = "gamified"

# STANDARDIZE ENGAGEMENT COLUMN NAMES ACROSS CONDITIONS
# Minimal and Gamified label these "Engagement Scale for Puzzle [...]"
# but Scaffolded labels the SAME questions "Step 3: Share Your Experience [...]"
# Rename all three to one common set of names before combining.

engagement_rename_map = {
    "felt_engaged": ["Engagement Scale for Puzzle [I felt engaged]",
                      "Step 3: Share Your Experience [I felt engaged]"],
    "enjoyed": ["Engagement Scale for Puzzle [I enjoyed completing it]",
                "Step 3: Share Your Experience [I enjoyed completing it]"],
    "would_again": ["Engagement Scale for Puzzle [I would do a similar task again]",
                     "Step 3: Share Your Experience [I would do a similar task again]"],
    "prior_exp": ["Engagement Scale for Puzzle [I've done many similar puzzles]",
                   "Step 3: Share Your Experience [I've done many similar puzzles]"],
}

def standardize_engagement_columns(df):
    for standard_name, possible_names in engagement_rename_map.items():
        for old_name in possible_names:
            if old_name in df.columns:
                df.rename(columns={old_name: standard_name}, inplace=True)
    return df

df_minimal = standardize_engagement_columns(df_minimal)
df_scaffolded = standardize_engagement_columns(df_scaffolded)
df_gamified = standardize_engagement_columns(df_gamified)

df = pd.concat([df_minimal, df_scaffolded, df_gamified], ignore_index=True)

# STEP 2: Convert Likert-text columns into numbers
likert_map = {
    "Disagree": 1, "Disagree ❌": 1,
    "Mid-Range": 2,
    "Agree": 3, "Agree ✔️": 3,
}

engagement_items = ["felt_engaged", "enjoyed", "would_again"]
prior_experience_item = "prior_exp"

for col in engagement_items:
    df[col + "_num"] = df[col].map(likert_map)

df["engagement_score"] = df[[c + "_num" for c in engagement_items]].mean(axis=1)
df["prior_experience"] = df[prior_experience_item].map(likert_map)

# STEP 3: Convert time buckets into numbers
time_map = {
    "Under 1 minute": 0.5,
    "1–3 minutes": 2,
    "3–5 minutes": 4,
    "More than 5 minutes": 6,
}
TIME_TEXT_COL = "How long did you spend on the puzzle?"
df["time_on_task"] = df[TIME_TEXT_COL].map(time_map)

# STEP 4: Parse the Score column (e.g. "50/60" -> 0.833)
def parse_score(x):
    if isinstance(x, str) and "/" in x:
        num, denom = x.split("/")
        return float(num) / float(denom)
    return None

df["score_pct"] = df["Score"].apply(parse_score)

# Final variable names
CONDITION_COL = "condition"
ENGAGEMENT_COL = "engagement_score"
TIME_COL = "time_on_task"
SCORE_COL = "score_pct"
PRIOR_EXP_COL = "prior_experience"

print(df.head())
print(df[CONDITION_COL].value_counts())  # confirm ~30 per group

# ONE-WAY ANOVA
def run_anova(dv_col, dv_label):
    groups = [df[df[CONDITION_COL] == c][dv_col].dropna()
              for c in df[CONDITION_COL].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n--- ANOVA: {dv_label} ---")
    print(f"F = {f_stat:.3f}, p = {p_val:.4f}")
    print("Significant (p < 0.05)" if p_val < 0.05 else "Not significant (p >= 0.05)")
    return f_stat, p_val

anova_engagement = run_anova(ENGAGEMENT_COL, "Engagement")
anova_score = run_anova(SCORE_COL, "Puzzle Accuracy (Score)")
anova_time = run_anova(TIME_COL, "Time on Task")

# TUKEY POST-HOC (only runs if the matching ANOVA was significant)
def run_tukey(dv_col, dv_label):
    print(f"\n--- Tukey HSD: {dv_label} ---")
    tukey = pairwise_tukeyhsd(endog=df[dv_col].dropna(),
                               groups=df.loc[df[dv_col].notna(), CONDITION_COL],
                               alpha=0.05)
    print(tukey)
    return tukey

if anova_engagement[1] < 0.05:
    tukey_engagement = run_tukey(ENGAGEMENT_COL, "Engagement")
if anova_score[1] < 0.05:
    tukey_score = run_tukey(SCORE_COL, "Puzzle Accuracy (Score)")
if anova_time[1] < 0.05:
    tukey_time = run_tukey(TIME_COL, "Time on Task")

# REGRESSION — prior experience predicting engagement
X = sm.add_constant(df[[PRIOR_EXP_COL]])
y = df[ENGAGEMENT_COL]
model = sm.OLS(y, X, missing='drop').fit()
print("\n--- Regression: Prior Experience -> Engagement ---")
print(model.summary())

# VISUALIZATIONS
sns.set_theme(style="whitegrid")

for col, ylabel, title, fname in [
    (ENGAGEMENT_COL, "Engagement Score", "Mean Engagement by Condition", "fig1_engagement_bar.png"),
    (SCORE_COL, "Score (proportion correct)", "Mean Puzzle Accuracy by Condition", "fig2_accuracy_bar.png"),
    (TIME_COL, "Time on Task", "Mean Time on Task by Condition", "fig3_time_bar.png"),
]:
    plt.figure(figsize=(7,5))
    sns.barplot(data=df, x=CONDITION_COL, y=col, errorbar='se', palette="pastel")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Condition")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

# Regression scatter: prior experience vs. engagement
plt.figure(figsize=(7,5))
sns.regplot(data=df, x=PRIOR_EXP_COL, y=ENGAGEMENT_COL,
            scatter_kws={"alpha":0.6}, line_kws={"color":"red"})
plt.title("Prior Experience vs. Engagement")
plt.xlabel("Prior Experience Rating")
plt.ylabel("Engagement Score")
plt.tight_layout()
plt.savefig("fig4_prior_experience_regression.png", dpi=300)
plt.show()

# Boxplot: within-condition spread in engagement (supports the
# individual-variation interpretation discussed in your Results/Discussion)
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x=CONDITION_COL, y=ENGAGEMENT_COL, palette="pastel")
sns.stripplot(data=df, x=CONDITION_COL, y=ENGAGEMENT_COL, color="black", alpha=0.4, jitter=True, size=4)
plt.title("Distribution of Engagement Scores by Condition")
plt.ylabel("Engagement Score")
plt.xlabel("Condition")
plt.tight_layout()
plt.savefig("fig5_engagement_boxplot.png", dpi=300)
plt.show()

# SUMMARY TABLE
summary = df.groupby(CONDITION_COL)[[ENGAGEMENT_COL, SCORE_COL, TIME_COL]].agg(['mean', 'std', 'count'])
print("\n--- Summary Table (for Results section) ---")
print(summary)
