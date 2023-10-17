import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import csv
from openpyxl import Workbook

# Percentage RMS calculation
def load_ecg(filename,bits):
    original =[]
    p_signal = []
    atr_sym = []
    atr_sample = []
    with open(f"C:/Users/ISURI/Documents/fyp/compression/CSVdata/p_signal_{filename}.csv", 'r') as file:
    # with open(f"C:/Users/ISURI/Documents/fyp/compression/CSVdata/p_signal_{filename}.csv", 'r') as file:
    
        csv_reader = csv.reader(file)
        for row in csv_reader:
            original.append(float(row[0]))
    original = np.array(original)

    with open(f"C:/Users/ISURI/Documents/fyp/compression/int_compressed_files/steps_{bits}/compressed_{filename}.txt", 'r') as file:
    # with open(f"C:/Users/ISURI/Documents/fyp/compression/CSVdata/p_signal_{filename}.csv", 'r') as file:
    
        csv_reader = csv.reader(file)
        for row in csv_reader:
            p_signal.append(float(row[0]))

    p_signal = np.array(p_signal)

    with open(f"C:/Users/ISURI/Documents/fyp/compression/CSVdata/atr_sample_{filename}.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            atr_sample.append((int)(float(row[0])))

    
    atr_sample = np.array(atr_sample)

    return original,p_signal, atr_sym, atr_sample


train_DS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test_DS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
steps = [10,9,8,7,6,5,3,2,1]
r = train_DS + test_DS

for s in steps:
  STEP = 1<< s

  writetofile = True
  writetofile = False

  if writetofile:
    wb = Workbook()
    ws = wb.active
    ws.append(["Record","TP","FP","FN","recall","acc","prec"])


  cumError = 0
  v1m1 = 0
  v2m1 = 0
  v1m = 0
  v2m = 0

  for record in r:
    original,p_signal, atr_sym, atr_sample = load_ecg(str(record), s)
    
    p_signal = p_signal * STEP

    plt.plot(p_signal/(1<<10)*5, color = 'orange', label = 'Quantized')
    
    
    b_coeff, a_coeff = butter(3, 0.15, btype='low')
    p_signal = filtfilt(b_coeff, a_coeff, p_signal.copy())
    
    p_signal = (p_signal/(1<<10)) * 5
    plt.plot(original, label = 'Original')
    plt.plot(p_signal, color = 'purple',label = 'Reconstructed')
    plt.legend()
    plt.show()
   
    print(record)

  if writetofile:
    ws.append(["", "","","", f"=average(e2:e{len(r)+1})",f"=average(f2:f{len(r)+1})",f"=average(g2:g{len(r)+1})"])
    wb.save(f"C:/Users/ISURI/Documents/fyp/compression/int_compressed_files/bits_vs_acc/bits_{s}.xlsx")
    # wb.save("Testing7/org_tol30_win250_gamma0.175_alpha0.05.xlsx")
  print("bits tested", s)