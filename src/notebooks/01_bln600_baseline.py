import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loader import *
from normalization import normalize_text_all
from metrics import cer

PAGES_TO_LOAD = 20

df = load_bln600_metadata()

for i in range(PAGES_TO_LOAD):
     page_id = df.iloc[i]["page-id"]
     ocr_s, gt_s = load_text_pair(page_id)
     
     ocr, gt = ocr_s.array[0], gt_s.array[0]
     ocr, gt = normalize_text_all(ocr, gt)
     
     _cer = cer(ocr, gt)
     
     print(f"Page ID: {page_id} has a CER of {_cer} between ocr and gt texts.")