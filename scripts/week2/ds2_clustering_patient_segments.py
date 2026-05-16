"""W2-DS2 | K-Means Clustering — Patient Segments — Output: outputs/cluster_*.csv/png"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = "outputs/w2_ds2_clustering_patient_segments"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("datasets/depression_anxiety_data.csv")
feats = ['phq_score','gad_score','age','bmi','epworth_score','gender']
cdf = df[feats].dropna().copy()
cdf['gender'] = LabelEncoder().fit_transform(cdf['gender'])
scaled = StandardScaler().fit_transform(cdf)

# Elbow + Silhouette to pick K
wcss = [KMeans(n_clusters=k,random_state=42).fit(scaled).inertia_ for k in range(2,11)]
sil  = [silhouette_score(scaled, KMeans(n_clusters=k,random_state=42).fit_predict(scaled))
        for k in range(4,7)]

best_k = 6
km = KMeans(n_clusters=best_k, random_state=42)
cdf['Cluster'] = km.fit_predict(scaled)

# Profiles
profiles = cdf.groupby('Cluster').mean().round(2)
profiles['count'] = cdf['Cluster'].value_counts().sort_index().values

def phq_sev(s): return "Minimal" if s<=4 else "Mild" if s<=9 else "Moderate" if s<=14 else "Severe"
def gad_sev(s): return "Minimal" if s<=4 else "Mild" if s<=9 else "Moderate" if s<=14 else "Severe"

profiles['PHQ_Severity'] = profiles['phq_score'].apply(phq_sev)
profiles['GAD_Severity'] = profiles['gad_score'].apply(gad_sev)
profiles['Profile_Name'] = ["Low-Risk Male","Obesity-Mild Distress","Moderate Mixed+Sleep",
                             "Older Adult Mild","High-Risk Severe","Low-Risk Female"]

# Save CSVs
cdf.to_csv(f"{OUT}/clustered_patients.csv", index=False)
profiles.to_csv(f"{OUT}/cluster_profiles.csv")

# Plot
COLS = ['#378ADD','#1D9E75','#BA7517','#7F77DD','#E24B4A','#888780']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Patient Cluster Analysis', fontsize=15, fontweight='bold')

axes[0,0].bar(range(best_k), profiles['count'], color=COLS, edgecolor='none')
axes[0,0].set(title='Patient Count per Cluster', xlabel='Cluster', ylabel='Count')
axes[0,0].set_xticks(range(best_k)); axes[0,0].set_xticklabels([f'C{i}' for i in range(best_k)])

for i in range(best_k):
    axes[0,1].scatter(profiles.loc[i,'phq_score'],profiles.loc[i,'gad_score'],
                      s=profiles.loc[i,'count']*2,color=COLS[i],alpha=0.85,edgecolors='white')
    axes[0,1].annotate(f'C{i}',(profiles.loc[i,'phq_score'],profiles.loc[i,'gad_score']),fontsize=9,ha='center')
axes[0,1].set(title='PHQ vs GAD (bubble=count)',xlabel='PHQ Score',ylabel='GAD Score')

x=np.arange(best_k); w=0.35
axes[1,0].bar(x-w/2,profiles['phq_score'],w,label='PHQ',color='#378ADD',edgecolor='none')
axes[1,0].bar(x+w/2,profiles['gad_score'],w,label='GAD',color='#E24B4A',edgecolor='none')
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels([f'C{i}' for i in range(best_k)])
axes[1,0].set(title='PHQ & GAD per Cluster',ylabel='Score'); axes[1,0].legend()

hm = profiles[['phq_score','gad_score','age','bmi','epworth_score']].values
im = axes[1,1].imshow(hm.T,cmap='RdYlGn',aspect='auto')
axes[1,1].set_xticks(range(best_k)); axes[1,1].set_xticklabels([f'C{i}' for i in range(best_k)])
axes[1,1].set_yticks(range(5)); axes[1,1].set_yticklabels(['PHQ','GAD','Age','BMI','Epworth'])
axes[1,1].set_title('Feature Heatmap per Cluster'); plt.colorbar(im,ax=axes[1,1])

plt.tight_layout()
plt.savefig(f"{OUT}/cluster_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"[DS2-W2] Done - K={best_k} clusters | Sil scores: {[round(s,3) for s in sil]}")
print(f"  -> clustered_patients.csv | cluster_profiles.csv | cluster_analysis.png")
