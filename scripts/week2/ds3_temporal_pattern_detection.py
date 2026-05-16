"""W2-DS3 | Temporal Pattern Detection — Output: outputs/temporal_*.png/csv"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = "outputs/w2_ds3_temporal_pattern_detection"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("datasets/mental_health_proxy_timeseries.csv")
df = df[~df['year'].isin([2021,2022])].reset_index(drop=True)
df_annual = df[df['year'] >= 2002].copy()

COLS = ['burglary_worry_pct','car_crime_worry_pct','violent_crime_worry_pct',
        'feel_safe_male_pct','feel_safe_female_pct','feel_safe_all_pct']
LABELS = ['Burglary worry','Car crime worry','Violent crime worry',
          'Male feel safe','Female feel safe','All feel safe']

# Rolling averages (3-year)
for col in COLS:
    df[f'{col}_ra3'] = df[col].rolling(window=3, min_periods=2).mean().round(4)

# ── Plot 1: Rolling averages ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('CSEW Mental Health Proxy - 3-Year Rolling Averages', fontsize=13, fontweight='bold')

worry = COLS[:3]; safe = COLS[3:]
for col,lbl in zip(worry, LABELS[:3]):
    axes[0].plot(df['year'],df[col],'o--',alpha=0.35,markersize=4)
    axes[0].plot(df['year'],df[f'{col}_ra3'],lw=2.5,label=lbl)
axes[0].axvspan(2020.5,2022.5,color='red',alpha=0.08,label='COVID gap')
axes[0].set(ylabel='% high worry',title='Worry About Crime'); axes[0].legend(); axes[0].grid(alpha=0.3)

for col,lbl in zip(safe, LABELS[3:]):
    axes[1].plot(df['year'],df[col],'o--',alpha=0.35,markersize=4)
    axes[1].plot(df['year'],df[f'{col}_ra3'],lw=2.5,label=lbl)
axes[1].axvspan(2020.5,2022.5,color='red',alpha=0.08,label='COVID gap')
axes[1].set(ylabel='% feel safe',title='Feeling Safe Walking Alone After Dark')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/temporal_rolling_averages.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Chow Test ────────────────────────────────────────────────────────────────
def chow_test(series, years, break_year):
    mask = ~np.isnan(series); y=series[mask]; x=years[mask]
    def rss(xs,ys):
        if len(xs)<3: return np.nan
        sl,ic,*_=stats.linregress(xs,ys); return np.sum((ys-(sl*xs+ic))**2)
    m1=x<=break_year; m2=x>break_year
    r1,r2,rp=rss(x[m1],y[m1]),rss(x[m2],y[m2]),rss(x,y)
    if np.isnan(r1) or np.isnan(r2): return np.nan,np.nan
    k=2; n=mask.sum()
    F=((rp-r1-r2)/k)/((r1+r2)/(n-2*k))
    p=1-stats.f.cdf(F,k,n-2*k)
    return round(F,4),round(p,4)

BREAK = 2013
y_arr = df_annual['year'].values.astype(float)
results=[]
for col,lbl in zip(COLS,LABELS):
    F,p = chow_test(df_annual[col].values.astype(float), y_arr, BREAK)
    results.append({'Variable':lbl,'Break_Year':BREAK,'F_stat':F,'p_value':p,
                    'Significant':'YES' if isinstance(p,float) and p<0.05 else 'NO'})
res_df = pd.DataFrame(results)

# ── Plot 2: F-stat scan for each variable ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Chow Test — Structural Break Scan per Variable', fontsize=13, fontweight='bold')
axes = axes.flatten()
scan_years = df_annual['year'].values[3:-3]

best_breaks = []
for ax, col, lbl in zip(axes, COLS, LABELS):
    s = df_annual[col].values.astype(float)
    scan = [(yr,*chow_test(s,y_arr,yr)) for yr in scan_years]
    scan_df = pd.DataFrame(scan, columns=['year','F_stat','p_value'])
    best = scan_df.loc[scan_df['F_stat'].idxmax()]
    best_breaks.append({'Variable':lbl,'Best_Break_Year':int(best.year),
                        'F_stat':best.F_stat,'p_value':best.p_value})
    ax.plot(scan_df['year'],scan_df['F_stat'],marker='o',markersize=4,lw=1.8,color='steelblue')
    ax.axvline(best.year,color='red',linestyle='--',lw=1.5,label=f'Best: {int(best.year)}')
    ax.set(title=lbl,xlabel='Break Year',ylabel='F-stat'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/temporal_chow_scan.png", dpi=150, bbox_inches='tight')
plt.close()

# Save results
res_df.to_csv(f"{OUT}/temporal_chow_test_results.csv", index=False)
pd.DataFrame(best_breaks).to_csv(f"{OUT}/temporal_best_breaks.csv", index=False)

print("[DS3-W2] Done")
print(res_df.to_string(index=False))
print(f"  -> temporal_rolling_averages.png | temporal_chow_scan.png | temporal_chow_test_results.csv")
