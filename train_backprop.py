"""
Backpropagation — XOR Classification
2-layer MLP trained with backprop. Outputs weight gradients,
loss surface samples, and decision boundary evolution.
"""
import numpy as np, json
np.random.seed(1)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def sig_d(z): s=sigmoid(z); return s*(1-s)

# Architecture: 2 → 4 → 1
W1 = np.random.randn(2,4)*0.8
b1 = np.zeros((1,4))
W2 = np.random.randn(4,1)*0.8
b2 = np.zeros((1,1))
lr = 0.5
epochs = 100

loss_hist, acc_hist = [], []
grad_norm_hist = []
snapshots = []  # boundary snapshots

def forward(X):
    z1 = X@W1+b1; a1=sigmoid(z1)
    z2 = a1@W2+b2; a2=sigmoid(z2)
    return z1,a1,z2,a2

def predict(X):
    _,_,_,a2 = forward(X)
    return (a2>0.5).astype(float)

def get_boundary_grid():
    xs = np.linspace(-0.2,1.2,30)
    ys = np.linspace(-0.2,1.2,30)
    grid = []
    for xi in xs:
        row=[]
        for yi in ys:
            inp=np.array([[xi,yi]])
            _,_,_,a2=forward(inp)
            row.append(round(float(a2[0,0]),3))
        grid.append(row)
    return {"xs":[round(float(v),3) for v in xs],"ys":[round(float(v),3) for v in ys],"z":grid}

for ep in range(1, epochs+1):
    z1,a1,z2,a2 = forward(X)
    loss = float(np.mean((y-a2)**2))
    d2 = -(y-a2)*sig_d(z2)
    dW2 = a1.T@d2; db2=d2.sum(0,keepdims=True)
    d1 = d2@W2.T*sig_d(z1)
    dW1 = X.T@d1; db1=d1.sum(0,keepdims=True)
    gn = float(np.linalg.norm(dW1)+np.linalg.norm(dW2))
    W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2
    acc = float(np.mean(predict(X)==y))*100
    loss_hist.append(round(loss,6))
    acc_hist.append(round(acc,1))
    grad_norm_hist.append(round(gn,4))
    if ep in [1,10,25,50,75,100]:
        snapshots.append({"ep":ep,"boundary":get_boundary_grid(),"acc":round(acc,1)})

def test_xor(x0,x1):
    inp=np.array([[x0,x1]])
    _,_,_,a2=forward(inp)
    return {"prob":round(float(a2[0,0]),4),"pred":int(a2[0,0]>0.5)}

test_cases=[
    {"inputs":[0,0],"expected":0,**test_xor(0,0)},
    {"inputs":[0,1],"expected":1,**test_xor(0,1)},
    {"inputs":[1,0],"expected":1,**test_xor(1,0)},
    {"inputs":[1,1],"expected":0,**test_xor(1,1)},
]

result={
    "type":"backprop",
    "data_points":[{"x":float(xi[0]),"y":float(xi[1]),"label":int(yi[0])} for xi,yi in zip(X,y)],
    "loss_history":loss_hist,"acc_history":acc_hist,
    "grad_norm_history":grad_norm_hist,
    "snapshots":snapshots,
    "test_cases":test_cases,
    "epochs":epochs,
    "architecture":[2,4,1],
}
with open("/home/claude/nn_scripts/out_backprop.json","w") as f:
    json.dump(result,f)
print("Backprop done. Final acc:",acc_hist[-1])
