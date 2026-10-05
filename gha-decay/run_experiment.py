"""Paired GHA experiment; original course notebook remains untouched.

Run with the brain_ai Python interpreter. No downloads or extra packages.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import platform
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent / 'biai-mlp-0.1.0'
SOURCE = COURSE / 'mlp.ipynb'
sys.path.insert(0, str(COURSE))
from biai.reproducibility import seed_everything, parameter_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_course_definitions():
    """Read only the named definitions, never run course training cells."""
    nb = json.loads(SOURCE.read_text(encoding='utf-8'))
    wanted = {'ActivationMLP', 'generalized_hebbian_update'}
    nodes = []
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
                nodes.append(node)
    assert {n.name for n in nodes} == wanted
    ns = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['ActivationMLP'], ns['generalized_hebbian_update']


def rate(method, epoch):
    return 0.001 if method == 'fixed' else 0.001 * (1.0 - 0.9 * epoch / 9)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    for x, target in loader:
        scores = model(x)
        loss_sum += F.cross_entropy(scores, target, reduction='sum').item()
        correct += (scores.argmax(1) == target).sum().item()
        count += target.numel()
    return dict(loss=loss_sum / count, accuracy=correct / count, count=count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise SystemExit('Output already contains a run. Use --output with a new directory.')
    torch.set_num_threads(4)
    seed_everything(0, deterministic=True)
    Model, update = load_course_definitions()
    raw = datasets.MNIST(COURSE / 'data', train=True, download=False,
                         transform=transforms.ToTensor())
    test_raw = datasets.MNIST(COURSE / 'data', train=False, download=False)
    # ToTensor on uint8 grayscale images is exactly float32 / 255.
    x = raw.data.flatten(1).to(torch.float32).div_(255)
    test_x = test_raw.data.flatten(1).to(torch.float32).div_(255)
    for i in range(0, len(raw), 137):
        assert torch.equal(x[i], raw[i][0].flatten())
    indices = torch.randperm(len(raw), generator=torch.Generator().manual_seed(0))
    train_set = Subset(TensorDataset(x, raw.targets), indices[6000:].tolist())
    val_set = Subset(TensorDataset(x, raw.targets), indices[:6000].tolist())
    test_set = TensorDataset(test_x, test_raw.targets)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    protocol = dict(
        design='paired comparison; fixed split seed 0; initialization/order seeds 0,1,2',
        source=str(SOURCE), source_sha256=sha(SOURCE),
        data_sha256={p.name: sha(p) for p in sorted((COURSE / 'data/MNIST/raw').iterdir()) if p.is_file()},
        split_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
        train_n=54000, validation_n=6000, test_n=10000, batch_size=64,
        epochs=10, seeds=[0, 1, 2], hidden_units=128, bias=False,
        activation='ReLU; GHA uses pre-ReLU linear activity',
        preprocessing='float32 / 255; no centering; tensor cache equality checked against ToTensor',
        classifier_optimizer='SGD, lr=0.05, momentum=0, weight_decay=0',
        feature_rates={m: [rate(m, e) for e in range(10)] for m in ('fixed', 'linear_decay')},
        primary='paired difference in final (epoch 10) validation accuracy, decay minus fixed',
        secondary='final test accuracy; mean absolute off-diagonal weight cosine; late validation step change',
        late_change_definition='mean absolute successive change of validation accuracy within epochs 6-10; not a causal stability test',
        selection='one decay schedule fixed before running; no tuning; final epoch, no checkpoint selection',
        test_policy='evaluate each final model once, after all six training runs have completed',
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         device='cpu', threads=4, deterministic=True),
        time_started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    )
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    # Independent numerical check of the batched GHA formula on a small toy case.
    layer = nn.Linear(4, 3, bias=False)
    sample = torch.randn(5, 4)
    before = layer.weight.detach().clone()
    response = sample @ before.T
    expected = before.clone()
    for j in range(3):
        delta = torch.zeros(4)
        for a, y in zip(sample, response):
            delta += y[j] * a - sum(y[j] * y[k] * before[k] for k in range(j + 1))
        expected[j] += 0.001 * delta / len(sample)
    update(layer, sample, 0.001)
    torch.testing.assert_close(layer.weight, expected)
    assert abs(rate('linear_decay', 0) - .001) < 1e-12
    assert abs(rate('linear_decay', 9) - .0001) < 1e-12
    results = []
    for seed in [0, 1, 2]:
        for method in ['fixed', 'linear_decay']:
            seed_everything(seed, deterministic=True)
            model = Model(bias=False)
            initial_digest = parameter_digest(model)
            model.fc1.weight.requires_grad_(False)
            optimizer = torch.optim.SGD(model.fc2.parameters(), lr=.05)
            loader = DataLoader(train_set, batch_size=64, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            history = []
            start = time.perf_counter()
            for epoch in range(10):
                model.train()
                loss_sum, correct, count = 0., 0, 0
                order_digest = hashlib.sha256()
                for inputs, target in loader:
                    # Inputs are exact float32 pixel values; digest verifies paired batch order.
                    order_digest.update(inputs.numpy().tobytes())
                    order_digest.update(target.numpy().tobytes())
                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, target)
                    loss.backward()
                    optimizer.step()
                    update(model.fc1, inputs, rate(method, epoch))
                    loss_sum += loss.item() * len(target)
                    correct += (logits.argmax(1) == target).sum().item()
                    count += len(target)
                assert all(torch.isfinite(p).all() for p in model.parameters())
                validation = evaluate(model, val_loader)
                row = dict(epoch=epoch + 1, feature_lr=rate(method, epoch),
                           train_loss=loss_sum / count, train_accuracy=correct / count,
                           validation_loss=validation['loss'], validation_accuracy=validation['accuracy'],
                           order_sha256=order_digest.hexdigest())
                history.append(row)
                print(f"seed={seed} {method} epoch={epoch+1:02} val={100*validation['accuracy']:.2f}% lr={rate(method,epoch):.6f}", flush=True)
            directions = F.normalize(model.fc1.weight.detach(), dim=1)
            mask = ~torch.eye(128, dtype=torch.bool)
            cosine = (directions @ directions.T).abs()[mask].mean().item()
            late = [r['validation_accuracy'] for r in history[5:]]
            result = dict(seed=seed, method=method, initial_sha256=initial_digest,
                          final_sha256=parameter_digest(model), history=history,
                          mean_abs_feature_cosine=cosine,
                          late_validation_mean_absolute_step_pp=100*sum(abs(b-a) for a,b in zip(late,late[1:]))/4,
                          training_seconds=time.perf_counter()-start)
            checkpoint = out / f'{method}_seed{seed}.pt'
            torch.save(model.state_dict(), checkpoint)
            result['checkpoint_sha256'] = sha(checkpoint)
            results.append(result)
            (out / f'{method}_seed{seed}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        a, b = results[-2:]
        assert a['initial_sha256'] == b['initial_sha256']
        assert [r['order_sha256'] for r in a['history']] == [r['order_sha256'] for r in b['history']]
        assert a['history'][0] == b['history'][0], 'First epoch is identical in both schedules.'
    # All planned runs are complete before the test set is scored.
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    for result in results:
        model = Model(bias=False)
        model.load_state_dict(torch.load(out / f"{result['method']}_seed{result['seed']}.pt", weights_only=True))
        assert parameter_digest(model) == result['final_sha256']
        result['test'] = evaluate(model, test_loader)
        print(f"TEST seed={result['seed']} {result['method']} {100*result['test']['accuracy']:.2f}%", flush=True)
        (out / f"{result['method']}_seed{result['seed']}.json").write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert sha(SOURCE) == protocol['source_sha256']
    (out / 'results.json').write_text(json.dumps(dict(protocol=protocol, results=results), indent=2, ensure_ascii=False), encoding='utf-8')
    print('COMPLETE: six runs, paired initialization/order/first-epoch checks passed; original notebook unchanged.', flush=True)


if __name__ == '__main__':
    main()
