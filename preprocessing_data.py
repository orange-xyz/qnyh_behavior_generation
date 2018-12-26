in_path = r'H:\qnyh\my_code\qnyh_lstm\data\qnyh_raw_data\\'
out_path = r'H:\qnyh\my_code\qnyh_lstm\data\qnyh_data\\'
import sys
sys.path.append(r'H:\qnyh\my_code\qnyh_lstm\data\qnyh_raw_data\\')
import numpy as np
import pickle
import pandas as pd
from collections import Counter
sequence_length = 20

with open(in_path + r'comb_ids_69', 'rb') as f:
    comb_ids_69 = pickle.load(f)
with open(in_path + r'user_grades_69', 'rb') as f:
    user_grades_69 = pickle.load(f)
with open(in_path + r'user_profiles_69', 'rb') as f:
    user_profiles_69 = pd.read_pickle(f)

def text2code(text):
    all_text = []
    for item in text:
        all_text.extend(item)
    vocabulary = list(set(all_text))
    for i in range(len(text)):
        for j in range(len(text[i])):
            text[i][j] = vocabulary.index(text[i][j])
    return text

#comb_ids_69 = text2code(comb_ids_69)
#user_grades_69 = text2code(user_grades_69)
class_vocab = ['-1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '102', '103']
classes = [class_vocab.index(str(x)) for x in list(user_profiles_69['class'])]

with open(out_path + r'comb_ids_69', 'w') as f0:
    with open(out_path + r'user_grades_69', 'w') as f1:
        with open(out_path + r'user_class_69', 'w') as f2:
            for i in range(len(comb_ids_69)):
                for j in range(1, len(comb_ids_69[i]) // sequence_length):
                    f0.write(' '.join(np.array(comb_ids_69[i][sequence_length * (j - 1):sequence_length * j]).astype(str)) + '\n')
                    f1.write(' '.join(np.array(user_grades_69[i][sequence_length * (j - 1):sequence_length * j]).astype(str)) + '\n')
                    f2.write(' '.join(np.array([classes[i]] * 20).astype(str)) + '\n')

