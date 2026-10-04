import json
from pathlib import Path
import numpy as np
import pytest
from models import cnn_loss_grad, recurrent_loss_grad, predict_sequence, predict_digit


@pytest.mark.parametrize('kind', ['rnn', 'lstm'])
def test_recurrent_gradients(kind):
    rng = np.random.default_rng(10)
    h, gates = 2, 1 if kind == 'rnn' else 4
    p = {'w': rng.normal(0,.2,(3,h*gates)), 'b': rng.normal(0,.2,h*gates),
         'wy': rng.normal(0,.2,h), 'by':np.array(.1)}
    x, y = rng.normal(size=(2,3)), rng.normal(size=2)
    _, grads = recurrent_loss_grad(x,y,p,kind)
    for key in p:
        for idx in np.ndindex(p[key].shape):
            old = p[key][idx]
            p[key][idx] = old + 1e-5
            plus = recurrent_loss_grad(x,y,p,kind)[0]
            p[key][idx] = old - 1e-5
            minus = recurrent_loss_grad(x,y,p,kind)[0]
            p[key][idx] = old
            assert np.isclose(grads[key][idx],(plus-minus)/2e-5,atol=1e-6)


def test_convolution_gradients():
    rng = np.random.default_rng(5)
    p = {'filters':rng.normal(size=(8,3,3)), 'conv_b':rng.normal(size=8),
         'dense_w':rng.normal(0,.1,(72,10)), 'dense_b':np.zeros(10)}
    x, y = rng.random((2,8,8)), np.array([2,5])
    _, grads = cnn_loss_grad(x,y,p)
    for key, idx in [('filters',(2,1,1)),('conv_b',(2,)),('dense_w',(2,3)),('dense_b',(3,))]:
        old = p[key][idx]
        p[key][idx] = old + 1e-5
        plus = cnn_loss_grad(x,y,p)[0]
        p[key][idx] = old - 1e-5
        minus = cnn_loss_grad(x,y,p)[0]
        p[key][idx] = old
        assert np.isclose(grads[key][idx],(plus-minus)/2e-5,atol=1e-6)


@pytest.mark.parametrize('kind', ['rnn','lstm'])
def test_exported_prediction_uses_input(kind):
    data = json.loads(Path(__file__).with_name(f'out_{kind}.json').read_text())
    values = data['test_window' if kind == 'rnn' else 'test_input']
    expected = data['test_result' if kind == 'rnn' else 'test_output']
    assert np.isclose(predict_sequence(values,data),expected)
    assert not np.isclose(predict_sequence(values,data),predict_sequence(np.array(values)+1,data))
    for bad in [[1]*9,[np.nan]*10]:
        with pytest.raises(ValueError):
            predict_sequence(bad,data)


def test_cnn_export_matches_saved_samples():
    data = json.loads(Path(__file__).with_name('out_cnn.json').read_text())
    for case in data['test_cases']:
        pred, probs = predict_digit(case['pixels'],data)
        assert pred == case['pred']
        assert np.isclose(probs.sum(),1)
