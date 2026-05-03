"""
Perceptron — AND Gate Binary Classification
Trains a single perceptron on the AND logic gate.
Outputs decision boundary snapshots and weight history.
"""
import numpy as np, json
np.random.seed(0)

# AND gate data
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,0,0,1], dtype=float)

w = np.random.randn(2)*0.1
b = 0.0
lr = 0.3
epochs = 30

def step(z): return 1.0 if z >= 0 else 0.0
def predict_all(W,B):
    return np.array([step(np.dot(x,W)+B) for x in X])

loss_hist, acc_hist, w_snapshots = [], [], []

for ep in range(1, epochs+1):
    errors = 0
    for xi, yi in zip(X, y):
        z = np.dot(xi, w) + b
        yp = step(z)
        err = yi - yp
        if err != 0:
            w += lr * err * xi
            b += lr * err
            errors += 1
    preds = predict_all(w, b)
    acc = float(np.mean(preds == y))*100
    loss = float(np.mean((y - preds)**2))
    loss_hist.append(round(loss,4))
    acc_hist.append(round(acc,1))
    # save boundary: w[0]*x + w[1]*y + b = 0 => y = -(w[0]*x+b)/w[1]
    snap = {"ep": ep, "w": w.tolist(), "b": float(b), "loss": round(loss,4), "acc": round(acc,1)}
    w_snapshots.append(snap)

# decision boundary at 20 epoch intervals
boundaries = []
for snap in w_snapshots:
    if snap["ep"] in [1,5,10,20,30]:
        W,B = snap["w"], snap["b"]
        xs = [-0.3, 1.3]
        if abs(W[1]) > 1e-6:
            ys = [-(W[0]*x+B)/W[1] for x in xs]
        else:
            ys = [0,0]
        boundaries.append({"ep": snap["ep"], "xs": xs, "ys": [round(v,3) for v in ys], "acc": snap["acc"]})

# test function
def test_input(x0, x1):
    return int(step(w[0]*x0 + w[1]*x1 + b))

test_cases = [
    {"inputs":[0,0],"expected":0,"predicted":test_input(0,0)},
    {"inputs":[0,1],"expected":0,"predicted":test_input(0,1)},
    {"inputs":[1,0],"expected":0,"predicted":test_input(1,0)},
    {"inputs":[1,1],"expected":1,"predicted":test_input(1,1)},
]

result = {
    "type": "perceptron",
    "data_points": [{"x": float(xi[0]),"y": float(xi[1]),"label": int(yi)} for xi,yi in zip(X,y)],
    "boundaries": boundaries,
    "loss_history": loss_hist,
    "acc_history": acc_hist,
    "weights": {"w0": round(float(w[0]),4), "w1": round(float(w[1]),4), "b": round(float(b),4)},
    "test_cases": test_cases,
    "epochs": epochs,
}
with open("/home/claude/nn_scripts/out_perceptron.json","w") as f:
    json.dump(result, f)
print("Perceptron done. Final acc:", acc_hist[-1])
