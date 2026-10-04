"""NumPy forward and backward passes shared by training and the dashboard."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class Adam:
    def __init__(self, params, lr=0.01):
        self.params, self.lr, self.t = params, lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        for k, g in grads.items():
            g = g * min(1, 5 / (norm + 1e-12))
            self.m[k] = .9 * self.m[k] + .1 * g
            self.v[k] = .999 * self.v[k] + .001 * g ** 2
            self.params[k] -= self.lr * (self.m[k] / (1 - .9 ** self.t)) / (
                np.sqrt(self.v[k] / (1 - .999 ** self.t)) + 1e-8)


def cnn_forward(x, p):
    windows = sliding_window_view(x, (3, 3), axis=(1, 2))
    conv = np.einsum('nhwij,fij->nhwf', windows, p['filters']) + p['conv_b']
    relu = np.maximum(conv, 0)
    blocks = relu.reshape(len(x), 3, 2, 3, 2, 8)
    pool = blocks.max(axis=(2, 4))
    flat = pool.reshape(len(x), -1)
    logits = flat @ p['dense_w'] + p['dense_b']
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    return probs, (windows, conv, blocks, pool, flat)


def cnn_loss_grad(x, y, p):
    probs, (windows, conv, blocks, pool, flat) = cnn_forward(x, p)
    loss = -np.log(probs[np.arange(len(x)), y] + 1e-12).mean()
    dl = probs.copy()
    dl[np.arange(len(x)), y] -= 1
    dl /= len(x)
    grads = {'dense_w': flat.T @ dl, 'dense_b': dl.sum(axis=0)}
    dp = (dl @ p['dense_w'].T).reshape(pool.shape)
    mask = blocks == pool[:, :, None, :, None, :]
    mask = mask / mask.sum(axis=(2, 4), keepdims=True)
    dc = (mask * dp[:, :, None, :, None, :]).reshape(conv.shape) * (conv > 0)
    grads['filters'] = np.einsum('nhwij,nhwf->fij', windows, dc)
    grads['conv_b'] = dc.sum(axis=(0, 1, 2))
    return float(loss), grads


def recurrent_forward(x, p, kind):
    n, steps = x.shape
    h = np.zeros((n, p['wy'].shape[0]))
    c = np.zeros_like(h)
    cache = []
    for t in range(steps):
        inp = np.concatenate([x[:, t:t+1], h], axis=1)
        if kind == 'rnn':
            hn = np.tanh(inp @ p['w'] + p['b'])
            cache.append((inp, hn))
            h = hn
        else:
            f, i, g, o = np.split(inp @ p['w'] + p['b'], 4, axis=1)
            f, i, g, o = sigmoid(f), sigmoid(i), np.tanh(g), sigmoid(o)
            cn = f * c + i * g
            hn = o * np.tanh(cn)
            cache.append((inp, c, cn, f, i, g, o))
            h, c = hn, cn
    return h @ p['wy'] + p['by'], (h, cache)


def recurrent_loss_grad(x, y, p, kind):
    pred, (h, cache) = recurrent_forward(x, p, kind)
    err = pred - y
    dy = 2 * err / len(x)
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    grads['wy'], grads['by'] = h.T @ dy, np.array(dy.sum())
    dh = dy[:, None] * p['wy']
    dc = np.zeros_like(dh)
    for item in reversed(cache):
        if kind == 'rnn':
            inp, hn = item
            dz = dh * (1 - hn ** 2)
        else:
            inp, cp, cn, f, i, g, o = item
            tanhc = np.tanh(cn)
            dc = dc + dh * o * (1 - tanhc ** 2)
            dz = np.concatenate([dc * cp * f * (1-f), dc * g * i * (1-i),
                                 dc * i * (1-g ** 2), dh * tanhc * o * (1-o)], axis=1)
            dc = dc * f
        grads['w'] += inp.T @ dz
        grads['b'] += dz.sum(axis=0)
        dh = (dz @ p['w'].T)[:, 1:]
    return float(np.mean(err ** 2)), grads


def as_arrays(weights):
    return {k: np.asarray(v, dtype=float) for k, v in weights.items()}


def predict_sequence(values, data):
    x = np.asarray(values, dtype=float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError('Enter exactly ten finite numbers.')
    lo, hi = data['scale']
    pred, _ = recurrent_forward(((x-lo)/(hi-lo))[None, :], as_arrays(data['weights']), data['type'])
    return float(pred[0] * (hi - lo) + lo)


def predict_digit(pixels, data):
    x = np.asarray(pixels, dtype=float)
    if x.size != 64 or not np.isfinite(x).all() or np.any((x < 0) | (x > 16)):
        raise ValueError('Supply 64 pixel values between zero and sixteen.')
    probs, _ = cnn_forward(x.reshape(1, 8, 8) / 16, as_arrays(data['weights']))
    return int(probs[0].argmax()), probs[0]
