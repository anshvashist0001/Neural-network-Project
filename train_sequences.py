"""Train recurrent models on synthetic series with a chronological holdout."""
import json
from pathlib import Path
import numpy as np
from models import Adam, recurrent_forward, recurrent_loss_grad


def train(kind):
    rng = np.random.default_rng(7 if kind == 'rnn' else 42)
    n = 200 if kind == 'rnn' else 150
    t = np.linspace(0, 4*np.pi, n) if kind == 'rnn' else np.arange(n)
    series = (np.sin(t) + .05*rng.normal(size=n) if kind == 'rnn' else
              100 + .3*t + 12*np.sin(t/8) + 8*np.sin(t/20) + 3*rng.normal(size=n))
    cutoff = int(n * .8)
    lo, hi = series[:cutoff].min(), series[:cutoff].max()
    scaled = (series - lo)/(hi-lo)
    x = np.array([scaled[i-10:i] for i in range(10, n)])
    y = scaled[10:]
    split = cutoff - 10
    h = 8 if kind == 'rnn' else 12
    gates = 1 if kind == 'rnn' else 4
    p = {'w': rng.normal(0, .2, (h+1, h*gates)), 'b': np.zeros(h*gates),
         'wy': rng.normal(0, .2, h), 'by': np.array(0.)}
    if kind == 'lstm':
        p['b'][:h] = 1
    opt = Adam(p, lr=.01)
    loss_hist, r2_hist = [], []
    epochs = 250
    for _ in range(epochs):
        loss, grads = recurrent_loss_grad(x[:split], y[:split], p, kind)
        opt.step(grads)
        pred, _ = recurrent_forward(x[split:], p, kind)
        r2 = 1 - np.sum((pred-y[split:])**2)/np.sum((y[split:]-y[split:].mean())**2)
        loss_hist.append(loss)
        r2_hist.append(float(r2 * 100))
    pred, _ = recurrent_forward(x, p, kind)
    pred = pred*(hi-lo)+lo
    last = list(scaled[-10:])
    horizon = 20 if kind == 'rnn' else 15
    forecast = []
    for _ in range(horizon):
        yp, _ = recurrent_forward(np.array([last]), p, kind)
        forecast.append(float(yp[0]*(hi-lo)+lo))
        last = last[1:] + [float(yp[0])]
    test_input = series[-10:].tolist()
    result = {'type': kind, 'weights': {k:v.tolist() for k,v in p.items()},
              'scale': [float(lo), float(hi)], 'epochs': epochs,
              'loss_history': loss_hist, 'acc_history': r2_hist,
              'evaluation': {'split': 'chronological 80/20', 'train_end_index': cutoff,
                             'test_r2': r2_hist[-1]/100,
                             'test_mse': float(np.mean((series[cutoff:]-pred[split:])**2)),
                             'persistence_mse': float(np.mean((series[cutoff:]-series[cutoff-1:-1])**2))}}
    if kind == 'rnn':
        dt = t[1]-t[0]
        result.update(t=t[10:].tolist(), actual=series[10:].tolist(), predicted=pred.tolist(),
                      forecast_t=(t[-1]+dt*np.arange(1,horizon+1)).tolist(), forecast=forecast,
                      test_window=test_input, test_result=forecast[0])
    else:
        result.update(days_actual=list(range(10,n)), actual=series[10:].tolist(), predicted=pred.tolist(),
                      forecast_days=list(range(n,n+horizon)), forecast=forecast,
                      test_input=test_input, test_output=forecast[0], price_range=[float(lo),float(hi)])
    Path(__file__).with_name(f'out_{kind}.json').write_text(json.dumps(result), encoding='utf-8')
    print(f'{kind}: holdout R2={r2_hist[-1]/100:.4f}, MSE={result["evaluation"]["test_mse"]:.4f}')


if __name__ == '__main__':
    train('rnn')
    train('lstm')
