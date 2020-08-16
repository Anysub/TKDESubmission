import subprocess
import ujson as json
import numpy as np
import sys
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
runs=10
#Top k HAN, variant2； adjust train_per in helper.py
args = [
    'python3',
    'train.py',
    '--problem-path',
    '../../../LineGraphGCN/data/dblp2/',
    '--problem',
    'dblp',
    '--lr-init',
    '2e-4',
    '--weight-decay',
    '5e-4',
    '--optimizer',
    'adamw',
    '--lr-schedule',
    'cosine',
    '--factor',
    '0.1',
    '--lr-patience',
    '5',
    '--dropout',
    '0.5',
    '--input-dropout',
    '0.2',
    '--prep-class',
    'linear',
    '--n-train-samples',
    '1000,10',
    '--n-val-samples',
    '1000,10',
    '--prep-len',
    '256',
    '--in-node-len',
    '128',
    '--n-hid',
    '512',
    '--in-edge-len',
    '130',
    '--n-head',
    '4',
    '--output-dims',
    '128,128,32,32',
    '--n-layer',
    '1',
    '--tolerance',
    '100',
    '--epochs',
    '100',
    '--train-per',
    '0.3',
    '--batch-size',
    '64',
    '--val-batch-size',
    '64',
    '--concat-node',
    '--mpaggr-class',
    'gate',
]
print(args)
sys.stdout.flush()
test_acc = []
test_macro = []
for seed in range(0,runs):
    process = subprocess.Popen(args+['--seed',str(seed)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    text = process.communicate()[1]

    lines = text.decode().split('\n')
    
    correct = False
    for line in lines:
        if '{' not in line:
            continue
        print(line)
        line = json.loads(line)
        if 'test_metric' in line:
            correct = True
            test_acc.append(line['test_metric']['accuracy'])
            test_macro.append(line['test_metric']['macro'])
    if not correct:
        print(lines)
    sys.stdout.flush()
test_acc = np.asarray(test_acc)
test_macro = np.asarray(test_macro)
print('average acc for {} runs is : {}'.format(len(test_acc), np.average(test_acc)))
print('average macro for {} runs is : {}'.format(len(test_macro), np.average(test_macro)))




