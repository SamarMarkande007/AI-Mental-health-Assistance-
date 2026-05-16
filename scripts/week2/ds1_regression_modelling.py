"""W2-DS1 | Regression — GAD Score Predictors — Output: outputs/regression_*.csv/png"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

OUT = "outputs/w2_ds1_regression_modelling"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("datasets/depression_anxiety_data.csv")
df = df.dropna()
df = df.drop(['id','anxiety_severity','anxiousness','anxiety_diagnosis','anxiety_treatment',
              'who_bmi','depression_severity','sleepiness'], axis=1)

num_cols = ['school_year','age','bmi','phq_score','epworth_score']
cat_cols = ['gender','depressiveness','suicidal','depression_diagnosis','depression_treatment']

X = df.drop('gad_score', axis=1)
y = df['gad_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('pre', ColumnTransformer([('num',StandardScaler(),num_cols),('cat',OneHotEncoder(drop='first'),cat_cols)])),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
n, p = len(y_test), pipeline.named_steps['pre'].fit_transform(X_train).shape[1]
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

# Coefficients
feat_names = [f.replace('num__','').replace('cat__','')
              for f in pipeline.named_steps['pre'].get_feature_names_out()]
coef_df = pd.DataFrame({'Feature':feat_names,'Coefficient':pipeline.named_steps['model'].coef_})
coef_df = coef_df.sort_values('Coefficient',ascending=False).reset_index(drop=True)

# Metrics CSV
metrics = pd.DataFrame({'Metric':['MSE','R2','Adjusted_R2'],'Value':[round(mse,4),round(r2,4),round(adj_r2,4)]})
metrics.to_csv(f"{OUT}/regression_metrics.csv", index=False)
coef_df.to_csv(f"{OUT}/regression_coefficients.csv", index=False)

# Plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('GAD Score — Linear Regression Results', fontsize=13, fontweight='bold')

axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=30)
mn,mx=y_test.min(),y_test.max()
axes[0].plot([mn,mx],[mn,mx],'r--',lw=2)
axes[0].set(xlabel='Actual GAD Score',ylabel='Predicted GAD Score',
            title=f'Actual vs Predicted\nR²={r2:.3f} | Adj-R²={adj_r2:.3f}')

sns.barplot(x='Coefficient',y='Feature',data=coef_df,ax=axes[1],
            palette='coolwarm',hue='Feature',legend=False)
axes[1].set(title='Feature Coefficients',xlabel='Coefficient',ylabel='')
axes[1].axvline(0,color='k',lw=0.8,linestyle='--')

plt.tight_layout()
plt.savefig(f"{OUT}/regression_results.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[DS1-W2] Done — MSE={mse:.2f} | R²={r2:.2f} | Adj-R²={adj_r2:.2f}")
print(f"  -> regression_metrics.csv | regression_coefficients.csv | regression_results.png")
