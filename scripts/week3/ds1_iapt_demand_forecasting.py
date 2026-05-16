"""W3-DS1 | IAPT Demand Forecasting (Prophet + ARIMA) — Output: outputs/forecast_*.png/csv"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

OUT = "outputs/w3_ds1_iapt_demand_forecasting"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("datasets/referrals_past_data.csv")
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

split = df['Month'].max() - pd.DateOffset(months=6)
train, test = df[df['Month']<=split], df[df['Month']>split]

# ── Prophet ───────────────────────────────────────────────────────────────────
p_train = train[['Month','Referrals_Received']].rename(columns={'Month':'ds','Referrals_Received':'y'})
p_test  = test[['Month','Referrals_Received']].rename(columns={'Month':'ds','Referrals_Received':'y'})

m = Prophet(); m.fit(p_train)
future  = m.make_future_dataframe(periods=len(p_test)+3, freq='MS')
fc      = m.predict(future)
p_pred  = m.predict(m.make_future_dataframe(periods=len(p_test),freq='MS',include_history=False))['yhat'].values
p_mae   = mean_absolute_error(p_test['y'].values, p_pred)
p_rmse  = np.sqrt(mean_squared_error(p_test['y'].values, p_pred))

# ── ARIMA ─────────────────────────────────────────────────────────────────────
arima_tr = train.set_index('Month')['Referrals_Received']
arima_te = test.set_index('Month')['Referrals_Received']
ar_res   = ARIMA(arima_tr, order=(2,1,2)).fit()
ar_pred  = ar_res.predict(start=arima_te.index.min(), end=arima_te.index.max(), dynamic=False)
a_mae    = mean_absolute_error(arima_te.values, ar_pred.values)
a_rmse   = np.sqrt(mean_squared_error(arima_te.values, ar_pred.values))

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('IAPT Referral Forecast — Model Comparison', fontsize=13, fontweight='bold')

axes[0].plot(p_test['ds'],p_test['y'],'o-',label='Actual',color='steelblue',lw=2)
axes[0].plot(p_test['ds'],p_pred,'x--',label='Prophet',color='tomato',lw=2)
axes[0].set(title=f'Prophet  MAE={p_mae:.0f}  RMSE={p_rmse:.0f}',xlabel='Month',ylabel='Referrals')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].tick_params(axis='x',rotation=45)

axes[1].plot(arima_te.index,arima_te.values,'o-',label='Actual',color='steelblue',lw=2)
axes[1].plot(arima_te.index,ar_pred.values,'x--',label='ARIMA',color='seagreen',lw=2)
axes[1].set(title=f'ARIMA(2,1,2)  MAE={a_mae:.0f}  RMSE={a_rmse:.0f}',xlabel='Month',ylabel='Referrals')
axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].tick_params(axis='x',rotation=45)

plt.tight_layout()
plt.savefig(f"{OUT}/forecast_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: 3-month ahead forecast ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df['Month'],df['Referrals_Received'],label='Historical',color='steelblue',lw=2)
ax.plot(fc['ds'],fc['yhat'],label='Prophet Forecast',color='tomato',lw=2,linestyle='--')
ax.fill_between(fc['ds'],fc['yhat_lower'],fc['yhat_upper'],color='tomato',alpha=0.15,label='95% CI')
ax.set(title='Prophet — Historical + 3-Month Forecast',xlabel='Month',ylabel='Referrals')
ax.legend(); ax.grid(alpha=0.3); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f"{OUT}/forecast_3month_ahead.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Save forecast CSV ─────────────────────────────────────────────────────────
fc_out = fc[['ds','yhat','yhat_lower','yhat_upper']].tail(3).rename(
    columns={'ds':'Month','yhat':'Forecast','yhat_lower':'CI_Lower','yhat_upper':'CI_Upper'})
fc_out.to_csv(f"{OUT}/forecast_3month_values.csv", index=False)

metrics = pd.DataFrame({'Model':['Prophet','ARIMA'],'MAE':[round(p_mae,2),round(a_mae,2)],
                        'RMSE':[round(p_rmse,2),round(a_rmse,2)]})
metrics.to_csv(f"{OUT}/forecast_metrics.csv", index=False)

print(f"[DS1-W3] Done - Prophet MAE={p_mae:.0f} | ARIMA MAE={a_mae:.0f}")
print(f"  -> forecast_model_comparison.png | forecast_3month_ahead.png | forecast_metrics.csv")
