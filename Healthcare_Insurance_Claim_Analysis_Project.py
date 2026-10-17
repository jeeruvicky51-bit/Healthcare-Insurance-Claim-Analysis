# %% [markdown]
# # Healthcare Insurance Claim Analysis
# **Author:** Vicky Jeeru | Data Analyst & Power BI Developer  
# **Tools:** Python (Pandas, NumPy, Seaborn, Matplotlib), Power BI

# %% [markdown]
# ## 1. Introduction
#
# Insurers need to understand **which member characteristics drive claim costs** so they can price policies fairly and target prevention programs.
#
# This project analyzes **15,000 member records** using Pandas and exploratory data analysis (EDA) to answer:
#
# 1. Which factors (smoking, BMI, age, diabetes, exercise, sex, hereditary disease) are linked to higher claims?
# 2. How concentrated are claim costs within the member base?
# 3. Can members be grouped into simple risk tiers that a pricing or wellness team could act on?
#
# The cleaned data is exported at the end and visualized in an interactive **Power BI dashboard**.

# %% [markdown]
# ## 2. Data Loading and Inspection

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

# %%
# Update this path to the location of your dataset
DATA_PATH = "healthinsurance_insurance.csv"

df = pd.read_csv(DATA_PATH)
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# %%
df.head(10)

# %%
df.info()

# %%
df.describe().T

# %% [markdown]
# **Column guide:** `claim` is the target amount we analyze. `smoker`, `diabetes` and `regular_ex` are 0/1 flags. `bloodpressure` appears to be a diastolic-style reading (average about 69).

# %% [markdown]
# ## 3. Data Cleaning

# %% [markdown]
# ### 3.1 Missing values and duplicates

# %%
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])
print(f"\nMissing share -> age: {missing['age']/len(df):.1%}, bmi: {missing['bmi']/len(df):.1%}")
print(f"Exact duplicate rows: {df.duplicated().sum():,}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
sns.boxplot(x=df["age"], ax=axes[0], color="#13A89E");  axes[0].set_title("Boxplot of Age")
sns.boxplot(x=df["bmi"], ax=axes[1], color="#E4572E");  axes[1].set_title("Boxplot of BMI")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Handling missing values:** `age` (396 missing) and `bmi` (956 missing) are filled with each column's **own median**. The median is used instead of the mean because both columns contain extreme values that would distort an average.
#
# > Each column is filled separately. Filling the whole DataFrame with a single value would copy the age median into the BMI column and create a false spike of members at one BMI value.

# %%
df["age"] = df["age"].fillna(df["age"].median())
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

print("Remaining missing values:", int(df.isnull().sum().sum()))
print(f"Median age: {df['age'].median():.0f} | Median BMI: {df['bmi'].median():.1f}")

# %% [markdown]
# ### 3.2 Data quality checks

# %%
print(f"Records with blood pressure = 0 (invalid): {(df['bloodpressure'] == 0).sum():,} "
      f"({(df['bloodpressure'] == 0).mean():.1%})")
print(f"Age range: {df['age'].min():.0f} to {df['age'].max():.0f}")
print(f"Claim range: ${df['claim'].min():,.0f} to ${df['claim'].max():,.0f}")
print(f"Mean claim ${df['claim'].mean():,.0f} vs median ${df['claim'].median():,.0f}  -> right-skewed")

# %% [markdown]
# **Findings:** about 5% of blood pressure values are 0, which is not physiologically possible, so blood pressure is treated as a low-reliability field. Claims are right-skewed (mean above median), so medians are reported alongside averages. Duplicate rows are flagged but kept, because repeated profiles are plausible; results are essentially unchanged if they are removed (smokers: about 20% of members and 48% of claims either way).

# %% [markdown]
# ### 3.3 Feature engineering

# %%
def bmi_category(bmi):
    if bmi < 18.5:   return "Underweight"
    elif bmi < 25:   return "Normal"
    elif bmi < 30:   return "Overweight"
    return "Obese"

df["bmi_category"] = df["bmi"].apply(bmi_category)
df["bmi_category"] = pd.Categorical(df["bmi_category"],
                                    ["Underweight", "Normal", "Overweight", "Obese"], ordered=True)

df["age_band"] = pd.cut(df["age"], bins=[17, 29, 39, 49, 64],
                        labels=["18-29", "30-39", "40-49", "50-64"])

df["smoker_label"] = df["smoker"].map({0: "Non-smoker", 1: "Smoker"})
df["diabetes_label"] = df["diabetes"].map({0: "No diabetes", 1: "Diabetes"})

df[["age", "age_band", "bmi", "bmi_category", "smoker_label"]].head()

# %% [markdown]
# ## 4. Exploratory Data Analysis

# %% [markdown]
# ### 4.1 How are claims distributed?

# %%
plt.figure(figsize=(9, 3.6))
sns.histplot(data=df, x="claim", hue="smoker_label", bins=45, multiple="layer",
             palette={"Non-smoker": "#13A89E", "Smoker": "#E4572E"}, alpha=0.9)
plt.axvline(df["claim"].mean(), color="#0B2545", ls="--")
plt.title("Claim distribution by smoking status"); plt.xlabel("Claim amount ($)")
plt.show()

# %% [markdown]
# The distribution has **two humps**. The right-hand hump is almost entirely smokers, which suggests one variable splits the member base into two very different populations. A single "average claim" would mislead pricing.

# %% [markdown]
# ### 4.2 Correlation with claim amount

# %%
num_cols = ["age", "weight", "bmi", "no_of_dependents", "smoker",
            "bloodpressure", "diabetes", "regular_ex", "claim"]

plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix"); plt.show()

df[num_cols].corr()["claim"].drop("claim").sort_values(ascending=False).round(2)

# %% [markdown]
# **Insights**
# * **Smoking has by far the strongest relationship with claims (0.77).**
# * Age (0.30) and BMI (0.20) show weaker positive relationships; diabetes is 0.14.
# * Blood pressure (0.01) shows essentially no relationship.
# * Regular exercise is slightly negative (-0.06).

# %% [markdown]
# ## 5. Claim Analysis by Key Factors

# %% [markdown]
# ### 5.1 Smoking status

# %%
smoker_summary = df.groupby("smoker_label")["claim"].agg(
    members="count", avg_claim="mean", median_claim="median", total_claims="sum").round(0)
smoker_summary["share_of_members_%"] = (smoker_summary["members"] / len(df) * 100).round(1)
smoker_summary["share_of_claims_%"] = (smoker_summary["total_claims"] / df["claim"].sum() * 100).round(1)
smoker_summary

# %%
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="smoker_label", y="claim", palette=["#13A89E", "#E4572E"])
plt.title("Claim amount by smoking status"); plt.xlabel(""); plt.ylabel("Claim ($)")
plt.show()

# %% [markdown]
# **Insights**
# * Smokers are about **20% of members but about 48% of all claim dollars**.
# * The average smoker claim (about $32.3K) is roughly **3.7x** the non-smoker average (about $8.7K).
# * Non-smokers have many high outliers, but they remain far below the typical smoker claim.

# %% [markdown]
# ### 5.2 BMI and smoking

# %%
bmi_summary = df.groupby("bmi_category", observed=True)["claim"].agg(
    members="count", avg_claim="mean", median_claim="median").round(0)
bmi_summary

# %%
plt.figure(figsize=(8, 4.5))
sns.barplot(data=df, x="bmi_category", y="claim", hue="smoker_label", errorbar=None,
            palette={"Non-smoker": "#13A89E", "Smoker": "#E4572E"})
plt.title("Average claim by BMI category and smoking status")
plt.xlabel("BMI category"); plt.ylabel("Average claim ($)"); plt.legend(title="")
plt.show()

df.pivot_table(index="bmi_category", columns="smoker_label", values="claim",
               aggfunc="mean", observed=True).round(0)

# %% [markdown]
# **Insights**
# * Obese members average about 50% more than normal-weight members overall.
# * The real story is the **interaction**: obese smokers average roughly **$41K, about 5x the claim of normal-weight non-smokers**. BMI moves non-smokers by only about $2K but moves smokers by about $18K.
# * The Underweight group is small (about 200 members), so its averages are less reliable.

# %% [markdown]
# ### 5.3 Age and smoking

# %%
age_tbl = df.pivot_table(index="age_band", columns="smoker_label", values="claim",
                         aggfunc="mean", observed=True).round(0)
display(age_tbl)

age_tbl.plot(marker="o", figsize=(7, 4), color=["#13A89E", "#E4572E"])
plt.title("Average claim by age band and smoking status")
plt.ylabel("Average claim ($)"); plt.xlabel("Age band"); plt.show()

print("Age vs claim correlation:", round(df["age"].corr(df["claim"]), 2))

# %% [markdown]
# **Insights**
# * Claims rise steadily with age in both groups. Among non-smokers they grow from about $4.7K (18-29) to about $13.5K (50-64), roughly **2.9x**.
# * A **young smoker (about $27.8K) costs more than a 50-64 year-old non-smoker (about $13.5K)**, so smoking status matters more than age for cost.

# %% [markdown]
# ### 5.4 Diabetes and smoking

# %%
diab_tbl = df.pivot_table(index="smoker_label", columns="diabetes_label", values="claim",
                          aggfunc="mean").round(0)
diab_tbl["diabetes_uplift_%"] = ((diab_tbl["Diabetes"] / diab_tbl["No diabetes"] - 1) * 100).round(0)
display(diab_tbl)

sns.barplot(data=df, x="smoker_label", y="claim", hue="diabetes_label", errorbar=None,
            palette={"No diabetes": "#B8C2CC", "Diabetes": "#0B2545"})
plt.title("Average claim by smoking status and diabetes"); plt.xlabel(""); plt.ylabel("Average claim ($)")
plt.legend(title=""); plt.show()

# %% [markdown]
# **Insights**
# * About 78% of members have diabetes, so the overall average hides a split.
# * Diabetes adds only about **10%** for non-smokers but about **52%** for smokers, so it matters mainly in combination with smoking. Diabetic smokers are the priority group for care management.

# %% [markdown]
# ### 5.5 Regular exercise

# %%
ex_tbl = df.pivot_table(index="smoker_label", columns="regular_ex", values="claim",
                        aggfunc=["count", "mean", "median"]).round(0)
ex_tbl

# %% [markdown]
# **Insights**
# * Among **smokers**, those who exercise regularly average about **$24.5K vs $34.2K** for those who do not (about 28% lower, only about 570 exercisers).
# * For non-smokers there is no clear benefit in the averages.
# * This is an observational pattern, so treat it as a hypothesis for a wellness-program pilot rather than proof.

# %% [markdown]
# ### 5.6 Sex

# %%
sex_tbl = df.groupby("sex").agg(members=("claim", "count"), avg_claim=("claim", "mean"),
                                smoker_rate=("smoker", "mean")).round(3)
display(sex_tbl)

df.pivot_table(index="smoker_label", columns="sex", values="claim", aggfunc="mean").round(0)

# %% [markdown]
# **Insights**
# * Men average about 11% more than women overall, but men also smoke more (about 23% vs 17%).
# * Within each smoking group the gap is small, so the sex difference is **largely a smoking-mix effect**.

# %% [markdown]
# ### 5.7 Hereditary disease (checking for confounding)

# %%
df["has_hereditary_disease"] = np.where(df["hereditary_diseases"] == "NoDisease", "No", "Yes")

hered = df.groupby("has_hereditary_disease").agg(
    members=("claim", "count"), avg_claim=("claim", "mean"), smoker_share=("smoker", "mean")).round(2)
display(hered)

(df[df["hereditary_diseases"] != "NoDisease"]
   .groupby("hereditary_diseases")["claim"].agg(["count", "mean"])
   .round(0).sort_values("mean"))

# %% [markdown]
# **Insights**
# * Members with a hereditary condition (about 1,000) average about **$39K vs $11.5K**.
# * However, **about 93% of them are smokers**, so the effect cannot be separated from smoking in this data. Checking for this kind of confounding avoids reporting a misleading driver.

# %% [markdown]
# ### 5.8 Blood pressure

# %%
valid_bp = df[df["bloodpressure"] > 0]
print("Correlation with claim (valid readings only):", round(valid_bp["bloodpressure"].corr(valid_bp["claim"]), 3))
valid_bp.groupby(pd.cut(valid_bp["bloodpressure"], [0, 60, 70, 80, 90, 130]), observed=True)["claim"].mean().round(0)

# %% [markdown]
# **Insight:** blood pressure shows no meaningful link to claims, and a share of readings are invalid zeros, so it is excluded from segmentation.

# %% [markdown]
# ## 6. Cost Concentration and Risk Tiers

# %% [markdown]
# ### 6.1 How concentrated are claim costs?

# %%
sorted_claims = df["claim"].sort_values(ascending=False).reset_index(drop=True)
cum_share = sorted_claims.cumsum() / sorted_claims.sum()

for pct in [0.05, 0.10, 0.20, 0.50]:
    print(f"Top {pct:.0%} of members -> {cum_share.iloc[int(len(df) * pct) - 1]:.1%} of claim dollars")

top10 = df.nlargest(int(len(df) * 0.10), "claim")
print(f"\nShare of smokers within the top 10% costliest members: {top10['smoker'].mean():.1%}")

# %% [markdown]
# The costliest **10% of members generate about 32% of claim dollars**, and **about 97% of them are smokers**.

# %% [markdown]
# ### 6.2 Simple risk tiers

# %%
conditions = [
    (df["smoker"] == 1) & (df["bmi"] >= 30),
    (df["smoker"] == 1),
]
df["risk_tier"] = np.select(conditions, ["1 - Critical (smoker, BMI 30+)", "2 - High (other smokers)"],
                            default="3 - Standard (non-smokers)")

tiers = df.groupby("risk_tier").agg(members=("claim", "count"),
                                    avg_claim=("claim", "mean"),
                                    total_claims=("claim", "sum"))
tiers["members_%"] = (tiers["members"] / len(df) * 100).round(1)
tiers["claims_%"] = (tiers["total_claims"] / df["claim"].sum() * 100).round(1)
tiers["avg_claim"] = tiers["avg_claim"].round(0)
tiers[["members", "members_%", "claims_%", "avg_claim"]]

# %%
tiers[["members_%", "claims_%"]].plot(kind="bar", figsize=(8, 4), color=["#B8C2CC", "#E4572E"])
plt.title("Share of members vs share of claim dollars by risk tier")
plt.ylabel("%"); plt.xlabel(""); plt.xticks(rotation=0, fontsize=8); plt.show()

# %% [markdown]
# **Insight:** two columns, smoking status and BMI, are enough to isolate a **Critical tier (about 9% of members) that produces about 26% of claim dollars**, and a **High tier (about 11% of members) producing about 21%**. The remaining 80% of members (non-smokers) account for about 52% of claims.

# %% [markdown]
# ## 7. Export for Power BI

# %%
export_cols = ["age", "sex", "weight", "bmi", "hereditary_diseases", "no_of_dependents",
               "smoker", "city", "bloodpressure", "diabetes", "regular_ex", "job_title",
               "claim", "bmi_category", "age_band", "risk_tier"]

export_df = df[export_cols].copy()
export_df["bmi_category"] = export_df["bmi_category"].astype(str)
export_df["age_band"] = export_df["age_band"].astype(str)

export_df.to_excel("Healthcare_Data.xlsx", index=False)
export_df.to_csv("Healthcare_Data.csv", index=False)
print("Exported", len(export_df), "rows for the Power BI dashboard")

# %% [markdown]
# The Power BI dashboard (`Healthinsurance_PowerBI_Dashboard.pbix`) reads `Healthcare_Data.xlsx` and presents these findings as an interactive one-page report with KPI cards, claim breakdowns by age, BMI, diabetes, smoking, sex and hereditary disease, and slicers for smoker, diabetes, sex and regular exercise.

# %% [markdown]
# ## 8. Conclusion
#
# * **Smoking is the dominant cost driver:** about 20% of members account for about 48% of claim dollars, and the average smoker claim is about 3.7x the non-smoker average.
# * **BMI, age and diabetes amplify the smoking effect** rather than acting alone: obese smokers average roughly $41K versus about $8.7K for all non-smokers.
# * **Sex and hereditary disease look like drivers but are largely explained by smoking**, which is why checking for confounding matters before making recommendations.
# * **Costs are concentrated:** the costliest 10% of members generate about 32% of claims and are almost all smokers.
#
# ## 9. Recommendations
#
# 1. **Price on smoking status first**, with an extra loading for obese smokers.
# 2. **Target smoking-cessation and weight-management programs** at the Critical tier.
# 3. **Prioritize diabetic smokers** for care management.
# 4. **Pilot exercise incentives** for smokers.
#
# ## 10. Limitations and Future Scope
#
# * Observational data: patterns show association, not causation.
# * Blood pressure contains invalid zeros, and there are no claim dates, so time trends cannot be analyzed.
# * Future work: add DAX measures and a star schema in Power BI, add claim dates for trend analysis, and add a risk-tier slicer to the dashboard.
