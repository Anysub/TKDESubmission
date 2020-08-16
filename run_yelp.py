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
    '../../../LineGraphGCN/data/yelp/',
    '--problem',
    'yelp',
    '--lr-init',
    '1e-4',
    '--weight-decay',
    '5e-4',
    '--dropout',
    '0.5',
    '--prep-class',
    'linear',
    '--n-train-samples',
    '100,100',
    '--n-val-samples',
    '100,100',
    '--prep-len',
    '128',
    '--in-edge-len',
    '18',
    '--n-head',
    '8',
    '--output-dims',
    '128,128,32,32',
    '--n-layer',
    '1',
    '--tolerance',
    '30',
    '--train-per',
    '0.4',
    '--batch-size',
    '64',
    '--val-batch-size',
    '64',
    '--K',
    '2599',
    '--concat-node',
    '--optimizer',
    'adam',
    '--lr-schedule',
    'const',
    '--mpaggr-class',
    'attention',
]
print(args)
test_acc = []
test_macro = []
for seed in range(runs):
    process = subprocess.Popen(args+['--seed',str(seed)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    text = process.communicate()[1]

    lines = text.decode().split('\n')
    # print(lines)
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




