"""W3-DS3 | Geo-spatial Analysis - IAPT x IMD - Output: outputs/spatial_*.png/csv"""

import pandas as pd, numpy as np, geopandas as gpd, matplotlib.pyplot as plt
import matplotlib.patches as mpatches, os, warnings
warnings.filterwarnings('ignore')

OUT = "outputs/w3_ds3_geo_spatial_analysis"
os.makedirs(OUT, exist_ok=True)

# Load data
nhs    = pd.read_csv("datasets/nhstalkingtherapies_month_feb_2026_activity_performance.csv")
imd    = pd.read_csv("datasets/File_1_IoD2025 Index of Multiple Deprivation.csv")
lookup = pd.read_csv("datasets/LSOA_(2021)_to_SICBL_to_ICB_to_LAD_(April_2023)_Lookup_in_EN.csv")
icb_map= gpd.read_file("datasets/ICB_APR_2023_EN_BGC.shp")

# ── NHS service metrics by ICB ────────────────────────────────────────────────
nhs_sub = nhs[nhs['GROUP_TYPE']=='SubICB'].copy()
nhs_sub['MEASURE_VALUE_SUPPRESSED'] = pd.to_numeric(nhs_sub['MEASURE_VALUE_SUPPRESSED'], errors='coerce')
code_to_icb = lookup[['SICBL23CDH','ICB23NM']].drop_duplicates().rename(columns={'SICBL23CDH':'ORG_CODE1'})
nhs_sub = nhs_sub.merge(code_to_icb, on='ORG_CODE1', how='left')

metrics = {
    'Count_ReferralsReceived':             ('Referrals',       'sum'),
    'Count_AccessingServices':             ('Accessing',       'sum'),
    'Percentage_ReliableRecovery':         ('Recovery_Rate',   'mean'),
    'Mean_WaitAccessingServices':          ('Mean_Wait_Days',  'mean'),
    'Count_WaitingForAssessmentOver90days':('Waiting_Over90',  'sum'),
}
service = None
for measure,(col,agg) in metrics.items():
    sub = nhs_sub[nhs_sub['MEASURE_NAME']==measure].groupby('ICB23NM')['MEASURE_VALUE_SUPPRESSED'].agg(agg).reset_index()
    sub.columns=['ICB23NM',col]
    service = sub if service is None else service.merge(sub,on='ICB23NM',how='outer')
service['Access_Rate'] = (service['Accessing']/service['Referrals']*100).round(2)

# ── IMD → ICB ─────────────────────────────────────────────────────────────────
imd = imd.rename(columns={
    'LSOA code (2021)':'LSOA21CD',
    'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)':'IMD_Rank',
    'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)':'IMD_Decile'})
n = len(imd)
imd['IMD_Score'] = ((n+1-imd['IMD_Rank'])/n*100).round(4)
imd = imd[['LSOA21CD','IMD_Score','IMD_Rank','IMD_Decile']]

imd_icb = (imd.merge(lookup[['LSOA21CD','ICB23NM']],on='LSOA21CD',how='inner')
           .groupby('ICB23NM').agg(IMD_Score_Mean=('IMD_Score','mean'),
                                   IMD_Decile_Mean=('IMD_Decile','mean'),
                                   n_LSOAs=('LSOA21CD','count')).reset_index().round(4))

# ── Join + Unmet Need Score ───────────────────────────────────────────────────
final = service.merge(imd_icb, on='ICB23NM', how='inner')

def norm(s, invert=False):
    mn,mx=s.min(),s.max()
    n=(s-mn)/(mx-mn)*100 if mx!=mn else pd.Series(50.0,index=s.index)
    return (100-n) if invert else n

final['Unmet_Need_Score'] = (
    0.35*norm(final['IMD_Score_Mean']) +
    0.30*norm(final['Access_Rate'],   invert=True) +
    0.20*norm(final['Recovery_Rate'], invert=True) +
    0.15*norm(final['Mean_Wait_Days'])
).round(2)

q75_imd  = final['IMD_Score_Mean'].quantile(0.75)
final['underserved'] = (
    (final['IMD_Score_Mean'] >= q75_imd) &
    ((final['Referrals']    <= final['Referrals'].quantile(0.25)) |
     (final['Access_Rate']  <= final['Access_Rate'].quantile(0.25)) |
     (final['Recovery_Rate']<= final['Recovery_Rate'].quantile(0.25)))
)

# ── Spatial join ──────────────────────────────────────────────────────────────
geo = icb_map.to_crs('EPSG:27700').merge(final, on='ICB23NM', how='left')

# ── Plot 1: IMD map ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
geo.plot(ax=ax, column='IMD_Score_Mean', cmap='YlOrRd', linewidth=0.5, edgecolor='gray', legend=True)
for _,row in geo.nlargest(5,'IMD_Score_Mean').iterrows():
    cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
    name=row['ICB23NM'].replace('NHS ','').replace(' Integrated Care Board','').strip()
    ax.annotate(name,xy=(cx,cy),fontsize=6,color='darkred',ha='center',fontweight='bold',
                bbox=dict(facecolor='white',alpha=0.8,edgecolor='none',pad=1))
ax.set_title('Average IMD Score by ICB — England',fontsize=12,fontweight='bold'); ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/spatial_imd_map.png", dpi=150, bbox_inches='tight'); plt.close()

# ── Plot 2: Access rate map ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
geo.plot(ax=ax, column='Access_Rate', cmap='RdYlGn', linewidth=0.5, edgecolor='gray', legend=True)
for _,row in geo.nsmallest(5,'Access_Rate').iterrows():
    cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
    name=row['ICB23NM'].replace('NHS ','').replace(' Integrated Care Board','').strip()
    ax.annotate(name,xy=(cx,cy),fontsize=6,color='darkred',ha='center',fontweight='bold',
                bbox=dict(facecolor='white',alpha=0.8,edgecolor='none',pad=1))
ax.set_title('NHS Talking Therapies — Access Rate by ICB',fontsize=12,fontweight='bold'); ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/spatial_access_rate_map.png", dpi=150, bbox_inches='tight'); plt.close()

# ── Plot 3: Underserved areas ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
geo[~geo['underserved'].fillna(False)].plot(ax=ax,color='lightgreen',edgecolor='gray',lw=0.6,alpha=0.8)
geo[geo['underserved'].fillna(False)].plot(ax=ax,color='red',edgecolor='darkred',lw=0.8,alpha=0.9)
handles=[mpatches.Patch(color='red',label='High-Need / Low-Service'),
         mpatches.Patch(color='lightgreen',label='Adequately Served')]
ax.legend(handles=handles,loc='lower left')
ax.set_title('High-Need / Low-Service ICBs — England',fontsize=12,fontweight='bold'); ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/spatial_underserved_map.png", dpi=150, bbox_inches='tight'); plt.close()

# ── Save CSV ──────────────────────────────────────────────────────────────────
final.to_csv(f"{OUT}/spatial_icb_analysis.csv", index=False)

n_u = final['underserved'].sum()
print(f"[DS3-W3] Done — {len(final)} ICBs | {n_u} underserved identified")
print(f"  Avg access rate: {final['Access_Rate'].mean():.1f}% | Recovery: {final['Recovery_Rate'].mean():.1f}%")
print(f"  -> spatial_imd_map.png | spatial_access_rate_map.png | spatial_underserved_map.png | spatial_icb_analysis.csv")
