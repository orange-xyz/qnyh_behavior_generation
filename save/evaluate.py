import os
import sys
import numpy as np
sys.path.append(r'H:\qnyh\Texygen\\')
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim

bleu = [0 for x in range(4)]
for i in range(4):
	bleu[i] = Bleu(test_text = r'H:\qnyh\Texygen\save\test_file.txt', real_text = r'H:\qnyh\Texygen\save\qnyh_ref_comb.txt', gram=i + 2)
	result = bleu[i].get_bleu()
	print(result)