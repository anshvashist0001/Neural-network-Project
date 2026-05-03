"""
MSE Linear Regression — House Price Prediction
Trains a simple linear regression model (y = w*x + b) via gradient descent.
Outputs scatter data, prediction line, and loss history.
"""
import numpy as np, json, random
random.seed(42); np.random.seed(42)

# ── Sample Data: house size (sqft/100) vs price ($k)
sizes = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,32,35,38,40],dtype=float)
noise = np.random.normal(0,18,len(sizes))
prices = 2.8*sizes + 40 + noise  # ground truth: price ≈ 2.8*size + 40

# normalize
sx,sy = sizes.std(),prices.std()
mx,my = sizes.mean(),prices.mean()
xn = (sizes-mx)/sx
yn = (prices-my)/sy

# ── Gradient Descent
w,b = 0.0, 0.0
lr = 0.05
epochs = 60
loss_hist, w_hist, b_hist = [], [], []

for ep in range(1, epochs+1):
    yp = w*xn + b
    err = yp - yn
    mse = float(np.mean(err**2))
    dw = float(np.mean(err*xn))
    db = float(np.mean(err))
    w -= lr*dw; b -= lr*db
    loss_hist.append(round(mse,6))
    w_hist.append(round(float(w),4))
    b_hist.append(round(float(b),4))

# final predictions on original scale
xline = np.linspace(sizes.min(), sizes.max(), 100)
xline_n = (xline - mx)/sx
yline = (w*xline_n + b)*sy + my

# test function
def predict(sqft_hundreds):
    xn_v = (sqft_hundreds - mx)/sx
    return float((w*xn_v + b)*sy + my)

test_cases = [
    {"sqft_100": 15, "predicted_price_k": round(predict(15),1)},
    {"sqft_100": 25, "predicted_price_k": round(predict(25),1)},
    {"sqft_100": 35, "predicted_price_k": round(predict(35),1)},
]

result = {
    "type": "mse",
    "scatter": [{"x": float(s*100), "y": round(float(p),1)} for s,p in zip(sizes,prices)],
    "line": [{"x": float(x*100), "y": round(float(y),1)} for x,y in zip(xline,yline)],
    "loss_history": loss_hist,
    "final_w": round(float(w*sy/sx),4),
    "final_b": round(float(b*sy + my - w*sy*mx/sx),4),
    "final_loss": round(loss_hist[-1],6),
    "test_cases": test_cases,
    "epochs": epochs,
    "labels": {"x": "House Size (sqft)", "y": "Price ($k)"},
}
with open("/home/claude/nn_scripts/out_mse.json","w") as f:
    json.dump(result, f)
print("MSE done. Final loss:", loss_hist[-1])
