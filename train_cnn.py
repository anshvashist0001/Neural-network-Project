"""Train a convolutional network on sklearn's 8x8 handwritten digits."""
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from models import Adam, cnn_forward, cnn_loss_grad


def train():
    rng = np.random.default_rng(0)
    digits = load_digits()
    indices = np.arange(len(digits.target))
    train_idx, test_idx = train_test_split(indices, test_size=.2, random_state=0, stratify=digits.target)
    train_idx, val_idx = train_test_split(train_idx, test_size=.2, random_state=1, stratify=digits.target[train_idx])
    x, y = digits.images / 16, digits.target
    p = {'filters': rng.normal(0, np.sqrt(2/9), (8,3,3)), 'conv_b': np.zeros(8),
         'dense_w': rng.normal(0, .1, (72,10)), 'dense_b': np.zeros(10)}
    opt = Adam(p, .005)
    losses, accuracies = [], []
    best_loss, best = float('inf'), None
    epochs = 35
    for epoch in range(epochs):
        shuffled = rng.permutation(train_idx)
        total = 0
        for start in range(0,len(shuffled),64):
            batch = shuffled[start:start+64]
            loss, grad = cnn_loss_grad(x[batch], y[batch], p)
            opt.step(grad)
            total += loss*len(batch)
        probs, _ = cnn_forward(x[val_idx],p)
        val_loss = -np.log(probs[np.arange(len(val_idx)),y[val_idx]]+1e-12).mean()
        losses.append(total/len(train_idx))
        accuracies.append(float((probs.argmax(1)==y[val_idx]).mean()*100))
        if val_loss < best_loss:
            best_loss, best = val_loss, {k:v.copy() for k,v in p.items()}
        if (epoch+1)%10 == 0:
            print(f'epoch {epoch+1}: loss={losses[-1]:.4f}, validation accuracy={accuracies[-1]:.2f}%')
    p = best
    probs, cache = cnn_forward(x[test_idx],p)
    predictions = probs.argmax(1)
    sample_images = [{'pixels':digits.images[idx].ravel().tolist(), 'true':int(y[idx]),
                      'pred':int(predictions[i]), 'correct':bool(predictions[i]==y[idx])}
                     for i,idx in enumerate(test_idx[:5])]
    test_cases = [{'pixels':digits.images[idx].ravel().tolist(), 'true':int(y[idx]),
                   'pred':int(predictions[i]), 'correct':bool(predictions[i]==y[idx])}
                  for i,idx in enumerate(test_idx[:10])]
    result = {'type':'cnn', 'weights':{k:v.tolist() for k,v in p.items()},
              'epochs':epochs, 'loss_history':losses, 'acc_history':accuracies,
              'final_acc':float((predictions==y[test_idx]).mean()*100),
              'filters':p['filters'].tolist(), 'feature_maps':[cache[1][0,:,:,i].tolist() for i in range(4)],
              'sample_images':sample_images, 'test_cases':test_cases,
              'confusion_matrix':confusion_matrix(y[test_idx],predictions,labels=np.arange(10)).tolist(),
              'architecture':['Input 8x8','Conv 3x3 x8','ReLU','MaxPool 2x2','Flatten 72','Dense 10','Softmax'],
              'evaluation':{'train_samples':len(train_idx),'validation_samples':len(val_idx),
                            'test_samples':len(test_idx),'seed':0,'selection':'minimum validation cross-entropy'}}
    Path(__file__).with_name('out_cnn.json').write_text(json.dumps(result),encoding='utf-8')
    print(f'CNN holdout accuracy: {result["final_acc"]:.2f}%')


if __name__ == '__main__':
    train()
