import numpy as np
from collections import Counter
with open(r'H:\qnyh\my_code\qnyh_lstm\data\task_data\qnyh_task_data_classes', 'r') as f:
    lines = f.readlines()
lines = [line.replace('\n', '').split(' ') for line in lines]
vocab = []
for line in lines:
    vocab.append(line[0])
vocab = [x[0] for x in Counter(vocab).most_common()]
for i in range(len(lines)):
    lines[i] = [vocab.index(x) for x in lines[i]]
with open(r'H:\qnyh\my_code\qnyh_lstm\data\task_data\classes_vocab', 'w') as f:
    f.write(' '.join(np.array(vocab).astype(str)))
with open(r'H:\qnyh\my_code\qnyh_lstm\data\task_data\qnyh_task_data_classes', 'w') as f:
    for i in range(len(lines)):
        f.write(' '.join(np.array(lines[i]).astype(str)) + '\n')