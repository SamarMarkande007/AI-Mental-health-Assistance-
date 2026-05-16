"""W3-DS2 | Prescribing Anomaly Detection (Isolation Forest) — Output: outputs/anomaly_*.csv/png"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

OUT = "outputs/w3_ds2_prescribing_anomaly_detection"
os.makedirs(OUT, exist_ok=True)

df1 = pd.read_csv("datasets/spending-by-ccg-.csv")
df2 = pd.read_csv("datasets/org_details.csv")
df  = pd.merge(df1, df2, on=['row_id','date'], how='inner')
df.drop(columns=['row_name_y'], inplace=True)
df.rename(columns={'row_name_x':'row_name'}, inplace=True)

# Feature engineering
df['cost_per_patient']     = df['actual_cost'] / df['total_list_size']
df['items_per_patient']    = df['items']        / df['total_list_size']
df['quantity_per_patient'] = df['quantity']     / df['total_list_size']
df['cost_per_item']        = df['actual_cost']  / df['items']
df['quantity_per_item']    = df['quantity']     / df['items']
df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=['cost_per_patient','items_per_patient',
                                                          'quantity_per_patient','cost_per_item','quantity_per_item'])

FEATURES = ['cost_per_patient','items_per_patient','quantity_per_patient','cost_per_item','quantity_per_item']
X_scaled = StandardScaler().fit_transform(df[FEATURES])

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)
df['anomaly_score'] = model.decision_function(X_scaled)
df['anomaly_flag']  = model.predict(X_scaled)
df['anomaly_flag']  = df['anomaly_flag'].map({1:'Normal',-1:'Anomaly'})

# ── Plot 1: Score distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Prescribing Anomaly Detection — Isolation Forest', fontsize=13, fontweight='bold')

axes[0].hist(df[df['anomaly_flag']=='Normal']['anomaly_score'],bins=50,alpha=0.7,color='steelblue',label='Normal')
axes[0].hist(df[df['anomaly_flag']=='Anomaly']['anomaly_score'],bins=50,alpha=0.7,color='tomato',label='Anomaly')
axes[0].axvline(0,color='k',linestyle='--',lw=1.5,label='Threshold')
axes[0].set(title='Anomaly Score Distribution',xlabel='Anomaly Score',ylabel='Count'); axes[0].legend()

top20 = (df[df['anomaly_flag']=='Anomaly'].groupby('row_name')['anomaly_flag']
         .count().sort_values(ascending=False).head(20))
top20.plot(kind='bar',color='tomato',alpha=0.8,ax=axes[1],edgecolor='none')
axes[1].set(title='Top 20 CCGs — Most Anomalous Months',xlabel='',ylabel='Anomalous Months')
axes[1].tick_params(axis='x',rotation=45)

plt.tight_layout()
plt.savefig(f"{OUT}/anomaly_overview.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Heatmap top 10 CCGs ──────────────────────────────────────────────
top10 = df[df['anomaly_flag']=='Anomaly']['row_name'].value_counts().head(10).index
pivot = df[df['row_name'].isin(top10)].pivot_table(index='row_name',columns='date',values='anomaly_score')
fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot, cmap='RdYlGn', center=0, linewidths=0.1, linecolor='grey', ax=ax)
ax.set(title='Anomaly Score Heatmap — Top 10 Most Anomalous CCGs Over Time')
plt.tight_layout()
plt.savefig(f"{OUT}/anomaly_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Save CSV ──────────────────────────────────────────────────────────────────
out = df[['row_id','row_name','date','actual_cost','items','quantity','total_list_size',
          'cost_per_patient','items_per_patient','quantity_per_patient',
          'cost_per_item','quantity_per_item','anomaly_score','anomaly_flag']].copy()
out = out.sort_values('anomaly_score', ascending=True)
out.to_csv(f"{OUT}/anomaly_flags.csv", index=False)

n_anom = (df['anomaly_flag']=='Anomaly').sum()
print(f"[DS2-W3] Done - {len(df):,} records | {n_anom:,} anomalies flagged ({n_anom/len(df)*100:.1f}%)")
print(f"  -> anomaly_flags.csv | anomaly_overview.png | anomaly_heatmap.png")
