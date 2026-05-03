import numpy as np, json
np.random.seed(42)
N=150; t=np.arange(N,dtype=float)
price=100+0.3*t+12*np.sin(t/8)+8*np.sin(t/20)+np.random.randn(N)*3
pmin,pmax=price.min(),price.max()
pn=(price-pmin)/(pmax-pmin)
win=10; X_all=[]; y_all=[]
for i in range(len(pn)-win): X_all.append(pn[i:i+win]); y_all.append(pn[i+win])
X_all,y_all=np.array(X_all),np.array(y_all)
n_h=12
def rand(r,c): return np.random.randn(r,c)*0.05
Wf=rand(1+n_h,n_h); bf=np.ones(n_h)*0.5
Wi=rand(1+n_h,n_h); bi=np.zeros(n_h)
Wg=rand(1+n_h,n_h); bg=np.zeros(n_h)
Wo=rand(1+n_h,n_h); bo=np.zeros(n_h)
Wy=np.random.randn(n_h)*0.05; by2=0.0
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-15,15)))
def lstm_fwd(x_seq):
    h=np.zeros(n_h); c=np.zeros(n_h)
    for xt in x_seq:
        xh=np.concatenate([[xt],h])
        f=sigmoid(xh@Wf+bf); i=sigmoid(xh@Wi+bi); g=np.tanh(xh@Wg+bg); o=sigmoid(xh@Wo+bo)
        c=f*c+i*g; h=o*np.tanh(c)
    return float(np.dot(h,Wy)+by2), h, c
lr=0.005; epochs=70; loss_hist=[]; acc_hist=[]
for ep in range(1,epochs+1):
    total=0; idx=np.random.permutation(len(X_all))[:30]
    for i in idx:
        yp,h,c=lstm_fwd(X_all[i]); err=yp-y_all[i]; loss=err**2; total+=loss
        Wy[:]-=lr*h*err*2; by2-=lr*err*2
    avg=total/30
    preds=np.array([lstm_fwd(xr)[0] for xr in X_all[:50]])
    ss_res=np.sum((y_all[:50]-preds)**2); ss_tot=np.sum((y_all[:50]-y_all[:50].mean())**2)
    r2=max(0,float(1-ss_res/(ss_tot+1e-9)))*100
    loss_hist.append(round(float(avg),6)); acc_hist.append(round(r2,2))
pred_norm=[lstm_fwd(xr)[0] for xr in X_all]
pred_price=[round(float(v*(pmax-pmin)+pmin),2) for v in pred_norm]
actual_price=[round(float(v*(pmax-pmin)+pmin),2) for v in y_all]
last_win=list(pn[-win:]); forecast_norm=[]
for _ in range(15):
    yp,_,_=lstm_fwd(last_win); forecast_norm.append(yp); last_win=last_win[1:]+[yp]
forecast_price=[round(float(v*(pmax-pmin)+pmin),2) for v in forecast_norm]
sample_recent=[round(float(v),2) for v in price[-10:].tolist()]
sample_pn=[(p-pmin)/(pmax-pmin) for p in sample_recent]
pred_next=round(float(lstm_fwd(sample_pn)[0]*(pmax-pmin)+pmin),2)
result={"type":"lstm","days_actual":list(range(win,N)),"actual":actual_price,"predicted":pred_price,
    "forecast_days":list(range(N,N+15)),"forecast":forecast_price,
    "loss_history":loss_hist,"acc_history":acc_hist,"epochs":epochs,
    "test_input":sample_recent,"test_output":pred_next,"price_range":[round(float(pmin),2),round(float(pmax),2)]}
with open("/home/claude/nn_scripts/out_lstm.json","w") as f: json.dump(result,f)
print("LSTM done. Final R²:",acc_hist[-1])
