import numpy as np, json
np.random.seed(7)
T=200; t=np.linspace(0,4*np.pi,T)
seq=np.sin(t)+0.05*np.random.randn(T)
win=10
X_all,y_all=[],[]
for i in range(len(seq)-win):
    X_all.append(seq[i:i+win]); y_all.append(seq[i+win])
X_all,y_all=np.array(X_all),np.array(y_all)
h_size=8
Wx=np.random.randn(h_size)*0.1
Wh=np.random.randn(h_size,h_size)*0.1
Wy=np.random.randn(h_size)*0.1
bh=np.zeros(h_size); by=0.0
lr=0.003; epochs=60; loss_hist=[]; acc_hist=[]

def rnn_forward(x_seq):
    h=np.zeros(h_size); hs=[h.copy()]
    for xt in x_seq:
        h=np.tanh(Wx*xt + h@Wh + bh); hs.append(h.copy())
    return float(np.dot(h,Wy)+by), hs

def train_step(x_seq, ytrue):
    global Wx,Wh,Wy,bh,by
    h=np.zeros(h_size); hs=[h.copy()]
    for xt in x_seq:
        h=np.tanh(Wx*xt + h@Wh + bh); hs.append(h.copy())
    yp=np.dot(h,Wy)+by; loss=(yp-ytrue)**2
    dh=Wy*(yp-ytrue)*2
    dWy=h*(yp-ytrue)*2; dby=float((yp-ytrue)*2)
    dWx=np.zeros_like(Wx); dWh=np.zeros_like(Wh); dbh_a=np.zeros_like(bh)
    for i in reversed(range(len(x_seq))):
        dt=(1-hs[i+1]**2)*dh
        dbh_a+=dt; dWx+=x_seq[i]*dt; dWh+=np.outer(hs[i],dt); dh=dt@Wh.T
    for g in [dWx,dWh,dWy,dbh_a]: np.clip(g,-1,1,out=g)
    Wx-=lr*dWx; Wh-=lr*dWh; Wy-=lr*dWy; bh-=lr*dbh_a; by-=lr*dby
    return float(loss)

for ep in range(1,epochs+1):
    total=0; idx=np.random.permutation(len(X_all))
    for i in idx[:40]: total+=train_step(X_all[i],y_all[i])
    avg=total/40
    preds=np.array([rnn_forward(xr)[0] for xr in X_all[:40]])
    ss_res=np.sum((y_all[:40]-preds)**2); ss_tot=np.sum((y_all[:40]-y_all[:40].mean())**2)
    r2=max(0,float(1-ss_res/(ss_tot+1e-9)))*100
    loss_hist.append(round(avg,6)); acc_hist.append(round(r2,2))

pred_seq=[round(float(rnn_forward(xr)[0]),4) for xr in X_all]
last_win=list(seq[-win:]); forecast=[]
for _ in range(20):
    yp,_=rnn_forward(last_win); forecast.append(round(float(yp),4)); last_win=last_win[1:]+[float(yp)]
t_plot=[round(float(v),4) for v in t[win:].tolist()]
actual_plot=[round(float(v),4) for v in y_all.tolist()]
forecast_t=[round(float(v),4) for v in np.linspace(t[-1],t[-1]+2*np.pi/5*20,20).tolist()]
result={"type":"rnn","t":t_plot[:150],"actual":actual_plot[:150],"predicted":pred_seq[:150],
    "forecast_t":forecast_t,"forecast":forecast,"loss_history":loss_hist,"acc_history":acc_hist,"epochs":epochs,
    "test_window":list(np.sin(np.linspace(0,np.pi,10)).round(3)),
    "test_result":round(float(rnn_forward(list(np.sin(np.linspace(0,np.pi,10))))[0]),4)}
with open("/home/claude/nn_scripts/out_rnn.json","w") as f: json.dump(result,f)
print("RNN done. Final R²:",acc_hist[-1])
