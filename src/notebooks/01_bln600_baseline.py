import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loader import *
from normalization import normalize_text_all
from metrics import cer

PAGES_TO_LOAD = 20

df = load_bln600_metadata()
example_printed = False

y_cer = []
x_page_id = []
for i in range(PAGES_TO_LOAD):
     page_id = df.iloc[i]["page-id"]
     ocr_s, gt_s = load_text_pair(page_id)
     
     before_ocr, before_gt = ocr_s.array[0], gt_s.array[0]
     ocr, gt = normalize_text_all(before_ocr, before_gt)
     
     if not example_printed:
          print(f"Before normalization ocr: {before_ocr}\nBefore normalization gt: {before_gt}")
          print(f"After normalization ocr: {ocr}\nAfter normalization gt: {gt}")
          example_printed = True
          
     _cer = cer(ocr, gt)
     x_page_id.append(i)
     y_cer.append(_cer)
     
     print(f"Page ID: {page_id} has a CER of {_cer} between ocr and gt texts.")

plt.figure(figsize=(10,6))
plt.plot(x_page_id, y_cer, marker="o")
plt.xlabel("Page ID")
plt.ylabel("CER")
plt.title("Plot for 20 pages loaded")
plt.show()