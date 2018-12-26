import numpy as np
out_grades_path = r'H:\qnyh\my_code\qnyh_lstm\data\task_data\\g_grades.txt'
out_classes_path = r'H:\qnyh\my_code\qnyh_lstm\data\task_data\\g_classes.txt'

grade_fixed = 20
class_fixed = 2
generated_num = 2048
sequence_length = 20

with open(out_grades_path, 'w') as f:
    gbuffer = ' '.join(np.repeat(grade_fixed, sequence_length).astype(str)) + '\n'
    for i in range(generated_num):
        f.write(gbuffer)

with open(out_classes_path, 'w') as f:
    cbuffer = ' '.join(np.repeat(class_fixed, sequence_length).astype(str)) + '\n'
    for i in range(generated_num):
        f.write(cbuffer)