# healthcare-insurance-claim-analysis
Healthcare insurance claim analysis using Python (Pandas, NumPy), exploratory data analysis, and Power BI dashboarding.



![Healthcare Insurance Claim Analysis Power BI Dashboard](images/dashboard.png)





## 🎯 Business Problem

Insurers that price every member the same overcharge healthy customers and underprice high-risk ones. This project answers three questions:

1. Which member characteristics drive claim amounts?
2. How concentrated are claim costs across the member base?
3. How can members be grouped into simple risk tiers to guide pricing and prevention programs?

## ⚡ Headline Results

| $201M | 3.7x | 47.8% | 31.8% |
|:---:|:---:|:---:|:---:|
| total claims analyzed (15,000 members) | higher average claim for smokers ($32.3K vs $8.7K) | of all claim dollars come from the 19.8% of members who smoke | of claim dollars come from the costliest 10% of members (97% are smokers) |

> **💡 Key insight:** Smoking status dominates everything else. **Smokers who are also obese average $41,086 per claim, 4.7x the $8,731 average for non-smokers.** Age, BMI and diabetes matter mainly because they *amplify* the smoking effect.

## 📊 Key Findings

### 1. Smoking is the #1 cost driver
Smokers are 19.8% of members but 47.8% of claim dollars. Smoking has a 0.77 correlation with claim amount, versus 0.30 for age and 0.20 for BMI.

<p align="center"><img src="images/share.png" width="650" alt="Share of members vs share of claims by smoking status"></p>

### 2. BMI raises risk, and smoking multiplies it
Obese smokers average **$41.1K**, 5.4x the claims of normal-weight non-smokers ($7.6K).

<p align="center"><img src="images/bmi_smoker.png" width="650" alt="Average claim by BMI category and smoking status"></p>

### 3. Age and diabetes: real effects, mostly through smoking
<p align="center">
  <img src="images/age.png" width="320" alt="Age by smoking">
  <img src="images/diab.png" width="320" alt="Diabetes by smoking">
</p>

| Signal | Result |
|---|---|
| Age (non-smokers) | $4.7K at 18-29 → $13.5K at 50-64 (**2.9x**) |
| Diabetes (smokers) | +52% ($34.3K vs $22.6K) |
| Diabetes (non-smokers) | only +10% ($8.9K vs $8.1K) |
| Regular exercise (smokers) | -28% ($24.5K vs $34.2K), observational, n = 574 |
| Sex | +11% for men overall, explained mostly by a higher smoking rate (22.8% vs 16.9%) |
| Hereditary disease | $39.4K vs $11.5K, but 93% of those members smoke, so the effects are confounded |

### 4. Claim costs are highly concentrated
The costliest 5% of members generate 17.4% of claim dollars, the top 10% generate **31.8%** and the top 20% generate 51.6%.

<p align="center">
  <img src="images/conc.png" width="320" alt="Cost concentration curve">
  <img src="images/corr.png" width="320" alt="Correlation with claim amount">
</p>

## 🧩 Risk Tiers

Two columns (smoking status and BMI) are enough to build practical tiers with one Pandas `np.select` rule:

<p align="center"><img src="images/tiers.png" width="380" alt="Risk tiers: share of members vs claims"></p>

| Tier | Rule | Members | Claim $ | Avg claim |
|---|---|---|---|---|
| **Critical** | Smoker, BMI 30+ | 8.6% | 26.3% | $41.1K |
| **High** | Other smokers | 11.2% | 21.4% | $25.6K |
| **Standard** | Non-smokers | 80.2% | 52.2% | $8.7K |

```python
conditions = [(df["smoker"] == 1) & (df["bmi"] >= 30),
              (df["smoker"] == 1)]
df["risk_tier"] = np.select(conditions,
    ["1 - Critical (smoker, BMI 30+)", "2 - High (other smokers)"],
    default="3 - Standard (non-smokers)")
```

## 🧹 Data Preparation

| Step | What I did |
|---|---|
| Profiling | Types, ranges, outliers (boxplots), claim distribution (mean $13,401 vs median $9,546, right-skewed) |
| Missing values | Age 396 (2.6%) and BMI 956 (6.4%) filled with **each column's own median**, which is robust to outliers |
| Imputation audit | Filling the whole DataFrame with the age median had copied it into BMI; corrected so each column uses its own median |
| Feature engineering | `pd.cut` age bands, WHO BMI categories, smoking and diabetes labels, `np.select` risk tiers |
| Quality flags | 756 records with blood pressure 0 (excluded from segmentation); 1,096 exact duplicate rows flagged but kept, as results are unchanged without them |

<p align="center"><img src="images/dist.png" width="620" alt="Claim distribution split by smoking status"></p>

## 🛠️ Analyst Toolkit Demonstrated

| Skill | Where it was used |
|---|---|
| Pandas cleaning & feature engineering | Median imputation, duplicate / invalid-value checks, `pd.cut`, `np.select` |
| Group-level EDA | `groupby`, `pivot_table`, multi-factor comparisons (smoking x BMI, age, diabetes, exercise) |
| Statistics | Correlation analysis, medians vs means for skewed data, confounding checks |
| Visualization | Seaborn / Matplotlib charts and a Power BI dashboard with slicers |

## 📊 Power BI Dashboard

One-page interactive report (16 visuals) built on the cleaned dataset exported from the notebook:

- **KPI cards** for headline claim and member metrics
- **Six chart views** of claims by age category, BMI category, diabetes, smoker status, sex and hereditary disease
- **Four slicers** (smoker, diabetes, sex, regular exercise) that filter the whole page

<!-- Add a screenshot: save it as images/dashboard.png, then uncomment the next line -->
<!-- <p align="center"><img src="images/dashboard.png" width="700" alt="Power BI dashboard"></p> -->

## ✅ Recommendations

1. **Price on smoking status first**, with an additional loading for obese smokers.
2. **Target cessation and weight-management programs** at the Critical tier.
3. **Prioritize diabetic smokers** for care management.
4. **Pilot exercise incentives** for smokers.

## 🔍 Limitations

- Observational data: patterns show association, not causation.
- Blood pressure has invalid zeros, and there are no claim dates, so trends over time cannot be analyzed.
- Hereditary disease and smoking overlap heavily (93%), so their separate effects cannot be isolated.

## 🚀 Next Steps

- [ ] Add DAX measures (average claim per member, smoker share, tier share) and a star schema
- [ ] Add a risk-tier slicer and a cost-concentration visual to the dashboard
- [ ] Collect claim dates to analyze cost trends over time

## 📁 Repository Structure

```
Healthcare-Insurance-Claim-Analysis/
├── README.md
├── Healthcare_Insurance_Claim_Analysis_Project.ipynb
├── Healthinsurance_PowerBI_Dashboard.pbix
├── images/            # charts and dashboard screenshot
└── report/
    └── Healthcare_Insurance_Claim_Analysis_Report.pdf
```

## ▶️ How to Reproduce

```bash
pip install pandas numpy seaborn matplotlib openpyxl jupyter
jupyter notebook Healthcare_Insurance_Claim_Analysis_Project.ipynb
```
Set `DATA_PATH` in the first data-loading cell to your local copy of the dataset and run all cells. The last cell exports `Healthcare_Data.xlsx`, which the `.pbix` file reads (refresh the data source in Power BI Desktop).

## 👤 About Me

**Vicky Jeeru**: Data Analyst & Power BI Developer, Hyderabad, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vicky-jeeru-257a18281/)
[![Portfolio](https://img.shields.io/badge/Portfolio-13A89E?style=flat&logo=googlechrome&logoColor=white)](https://vickyjeeru.my.canva.site/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:jeeruvicky51@gmail.com)

*Open to Data Analyst and BI roles. If this project was useful, a ⭐ is appreciated.*
