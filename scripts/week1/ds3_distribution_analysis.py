"""W1-DS3 | Distribution Analysis — Output: outputs/dist_*.png (5 plots)"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = "datasets"
OUT = "outputs/w1_ds3_distribution_analysis"
os.makedirs(OUT, exist_ok=True)

FILE = f"{DATA_DIR}/apms-2014-ch-02-tabs.xls"

raw = {k: pd.read_excel(FILE, sheet_name=s, header=None, engine='xlrd') for k,s in [
    ('cisr_age_sex','2.1'),('cmd_age_sex','2.3'),('cmd_ethnicity','2.7'),
    ('cmd_employment','2.9 '),('cmd_smoking','2.12'),('cmd_region','2.11')]}

BANDS  = ['0-5','6-11','12-17','18+']
AGES   = ['16-24','25-34','35-44','45-54','55-64','65-74','75+','All']
DISORDERS = ['GAD','Depression','Phobias','OCD','Panic','CMD-NOS','Any CMD']
REGIONS_S = ['NE','NW','Yorks','E.Mids','W.Mids','East','London','SE','SW']
ETH   = ['White British','White Other','Black/Black Br.','Asian/Asian Br.','Mixed/Other']
EMP   = ['Employed FT','Employed PT','Unemployed','Econ. Inactive']
SMOKE = ['Non-smoker','Ex-smoker','1-14/day','15+/day']

# Parse tables
def cisr_for(sex, row_idx):
    rows = raw['cisr_age_sex'].iloc[row_idx, 1:9].values
    return pd.DataFrame(raw['cisr_age_sex'].iloc[[r for r in range(row_idx, row_idx+4)], 1:9].values.astype(float),
                        index=BANDS, columns=AGES)

r = raw['cisr_age_sex']
cisr_men   = pd.DataFrame(r.iloc[[6,7,10,11],  1:9].values.astype(float), index=BANDS, columns=AGES)
cisr_women = pd.DataFrame(r.iloc[[16,17,20,21],1:9].values.astype(float), index=BANDS, columns=AGES)
cisr_all   = pd.DataFrame(r.iloc[[26,27,30,31],1:9].values.astype(float), index=BANDS, columns=AGES)

cmd_age    = pd.DataFrame(raw['cmd_age_sex'].iloc[24:31,1:9].values.astype(float),  index=DISORDERS, columns=AGES)
cmd_men    = pd.DataFrame(raw['cmd_age_sex'].iloc[6:13, 1:9].values.astype(float),  index=DISORDERS, columns=AGES)
cmd_women  = pd.DataFrame(raw['cmd_age_sex'].iloc[15:22,1:9].values.astype(float),  index=DISORDERS, columns=AGES)
cmd_eth    = pd.DataFrame(raw['cmd_ethnicity'].iloc[25:32, 1:6].values.astype(float), index=DISORDERS, columns=ETH)
cmd_emp    = pd.DataFrame(raw['cmd_employment'].iloc[25:32, 1:5].values.astype(float), index=DISORDERS, columns=EMP)
cmd_smoke  = pd.DataFrame(raw['cmd_smoking'].iloc[25:32,  1:5].values.astype(float), index=DISORDERS, columns=SMOKE)
cmd_region = pd.DataFrame(raw['cmd_region'].iloc[25:32,   1:10].values.astype(float), index=DISORDERS, columns=REGIONS_S)

# ── Plot 1: CIS-R by sex and age ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('CIS-R Score Distribution by Sex - APMS 2014', fontsize=13, fontweight='bold')
x = np.arange(4); w = 0.3
for i,(sex,col,df_) in enumerate([('Men','steelblue',cisr_men),('Women','tomato',cisr_women)]):
    axes[0].bar(x+i*w, df_['All'].values, w, label=sex, color=col, alpha=0.9, edgecolor='k', lw=0.5)
axes[0].set_xticks(x+w/2); axes[0].set_xticklabels(BANDS)
axes[0].set(xlabel='CIS-R Band', ylabel='%', title='Band Prevalence by Sex'); axes[0].legend()

c12 = lambda d: [d[a].iloc[2]+d[a].iloc[3] for a in AGES[:-1]]
xs = np.arange(7)
for df_, col, lbl in [(cisr_men,'steelblue','Men'),(cisr_women,'tomato','Women'),(cisr_all,'orange','All')]:
    axes[1].plot(xs, c12(df_), 'o-', color=col, lw=2, ms=7, label=lbl)
axes[1].set_xticks(xs); axes[1].set_xticklabels(AGES[:-1], rotation=30)
axes[1].set(ylabel='CIS-R ≥12 (%)', title='Clinical-Level CMD by Age & Sex'); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/dist_cisr_sex_age.png", dpi=150, bbox_inches='tight'); plt.close()

# ── Plot 2: CMD heatmap + sex bar ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('CMD Disorder Prevalence - APMS 2014', fontsize=13, fontweight='bold')
mat = cmd_age.iloc[:6,:7].values.astype(float)
im = axes[0].imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=14)
axes[0].set_xticks(range(7)); axes[0].set_xticklabels(AGES[:-1], rotation=30)
axes[0].set_yticks(range(6)); axes[0].set_yticklabels(DISORDERS[:6])
for i in range(6):
    for j in range(7):
        axes[0].text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=8,
                     color='white' if mat[i,j]>8 else 'black',fontweight='bold')
axes[0].set_title('Prevalence by Age Group (%)'); plt.colorbar(im,ax=axes[0],fraction=0.04)
x=np.arange(6); w=0.35
axes[1].bar(x-w/2,cmd_men.iloc[:6,-1].values,  w,label='Men',  color='steelblue',alpha=0.9,edgecolor='k',lw=0.5)
axes[1].bar(x+w/2,cmd_women.iloc[:6,-1].values,w,label='Women',color='tomato',   alpha=0.9,edgecolor='k',lw=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(DISORDERS[:6],rotation=15)
axes[1].set(ylabel='Prevalence (%)',title='Prevalence by Sex'); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/dist_cmd_heatmap_sex.png",dpi=150,bbox_inches='tight'); plt.close()

# ── Plot 3: Any CMD across segments ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Any CMD Across Population Segments - APMS 2014', fontsize=13, fontweight='bold')
for ax, vals, labels, title in [
    (axes[0], cmd_eth.loc['Any CMD'].values,   ETH,   'By Ethnicity'),
    (axes[1], cmd_emp.loc['Any CMD'].values,   EMP,   'By Employment'),
    (axes[2], cmd_smoke.loc['Any CMD'].values, SMOKE, 'By Smoking Status')]:
    bars=ax.bar(range(len(labels)),vals,color=['steelblue','tomato','seagreen','orange','purple'][:len(labels)],
                edgecolor='k',lw=0.5,alpha=0.9)
    ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=20,ha='right')
    ax.set(ylabel='Any CMD (%)',title=title); ax.set_ylim(0,max(vals)*1.3)
plt.tight_layout(); plt.savefig(f"{OUT}/dist_any_cmd_segments.png",dpi=150,bbox_inches='tight'); plt.close()

# ── Plot 4: Regional CMD ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Regional CMD — APMS 2014', fontsize=13, fontweight='bold')
ra = cmd_region.loc['Any CMD'].sort_values()
bars=axes[0].barh(ra.index,ra.values,color='teal',edgecolor='w',lw=0.5,alpha=0.9)
axes[0].bar_label(bars,fmt='%.1f%%',padding=4,fontsize=9)
axes[0].set(xlabel='Any CMD (%)',title='Any CMD by Region')
mat2=cmd_region.iloc[:6].values.astype(float)
im2=axes[1].imshow(mat2,cmap='Blues',aspect='auto')
axes[1].set_xticks(range(9)); axes[1].set_xticklabels(REGIONS_S,rotation=30)
axes[1].set_yticks(range(6)); axes[1].set_yticklabels(DISORDERS[:6])
for i in range(6):
    for j in range(9):
        axes[1].text(j,i,f'{mat2[i,j]:.1f}',ha='center',va='center',fontsize=7,
                     color='white' if mat2[i,j]>6 else 'black')
axes[1].set_title('Disorder x Region (%)'); plt.colorbar(im2,ax=axes[1],fraction=0.03)
plt.tight_layout(); plt.savefig(f"{OUT}/dist_regional_cmd.png",dpi=150,bbox_inches='tight'); plt.close()

# ── Plot 5: Temporal trend ────────────────────────────────────────────────────
years = [1993,2000,2007,2014]
trend = {'Men':[10.5,13.4,12.6,13.6],'Women':[17.7,19.2,20.1,21.4],'All':[14.1,16.4,16.4,17.6]}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CMD Temporal Trend 1993-2014 — APMS', fontsize=13, fontweight='bold')
for (sex,vals),col in zip(trend.items(),['steelblue','tomato','orange']):
    axes[0].plot(years,vals,'o-',color=col,lw=2.5,ms=9,label=sex)
axes[0].set(xticks=years,ylabel='CIS-R ≥12 (%)',title='Trend by Sex'); axes[0].legend(); axes[0].grid(alpha=0.3)
gap=[w-m for m,w in zip(trend['Men'],trend['Women'])]
bars=axes[1].bar(years,gap,color='purple',width=4,edgecolor='k',lw=0.6,alpha=0.9)
axes[1].bar_label(bars,fmt='%.1f pp',padding=3,fontsize=10)
axes[1].set(xticks=years,ylabel='pp gap (Women-Men)',title='Sex Gap Over Time')
plt.tight_layout(); plt.savefig(f"{OUT}/dist_temporal_trend.png",dpi=150,bbox_inches='tight'); plt.close()

print("[DS3] Done — 5 PNGs saved to outputs/")
print("  dist_cisr_sex_age.png | dist_cmd_heatmap_sex.png | dist_any_cmd_segments.png")
print("  dist_regional_cmd.png | dist_temporal_trend.png")
