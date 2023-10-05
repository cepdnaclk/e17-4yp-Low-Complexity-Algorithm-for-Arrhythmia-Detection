import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import csv
from openpyxl import Workbook

b = [0.0200833655642112, 0.0401667311284225, 0.0200833655642112]
a = [1, -1.56101807580072, 0.641351538057563]

b_2 = np.array([0.0232834986888859, 0.0698504960666577, 0.0698504960666577, 0.0232834986888859])
a_2 = np.array([1, -1.76004188034356, 1.18289361544819, -0.278285914287814])

# v1m1 = 0
# v2m1 = 0
# v1m = 0
# v2m = 0

def slidwind(signal):
  result = signal.copy()
  min_idx = 0
  max_idx = len(result) - 1 
  for i in range(len(result)-1):
    win = signal[max(min_idx, i): min(i+100,max_idx)]
    result[i] = max(win) - min(win)
  return result

def boundary(signal, alpha):
  upper = signal.copy()
  lower = signal.copy()
  upperlimit = -5
  lowerlimit = 5
  count_upper = 0
  count_lower = 0
  for i in range(len(signal)):
    if signal[i] > upperlimit:
      upperlimit = signal[i]
      count_upper = 0
    else:
      count_upper += 1
    if signal[i] < lowerlimit:
      lowerlimit = signal[i]
      count_lower = 0
    else:
      count_lower += 1
    
    upper[i] = upperlimit
    lower[i] = lowerlimit

    upperlimit -= (upperlimit - lowerlimit) * alpha
    lowerlimit += (upperlimit - lowerlimit) * alpha

  return upper, lower

def boundary2(signal, alpha):
  # upper = signal.copy()
  # upperlimit = -5

  # for i in range(len(signal)):
  #   if signal[i] > upperlimit:
  #     upperlimit = signal[i]
    
  #   upper[i] = upperlimit
  #   upperlimit -= (upperlimit) * alpha

  # return upper
  
  upper = signal.copy()
  lower = signal.copy()
  upperlimit = -5
  lowerlimit = 5
  count_upper = 0
  count_lower = 0
  for i in range(len(signal)):
    if signal[i] > upperlimit:
      upperlimit = signal[i]
      count_upper = 0
    else:
      count_upper += 1
    if signal[i] < lowerlimit:
      lowerlimit = signal[i]
      count_lower = 0
    else:
      count_lower += 1
    
    upper[i] = upperlimit
    lower[i] = lowerlimit

    upperlimit -= (upperlimit - lowerlimit) * alpha
    lowerlimit += (upperlimit - lowerlimit) * alpha * 3

  return upper, lower



STEP = 1<<4

def pwm(y_):
  global cumError
  y = y_*(1<<11)
  yresampled = ((int(y/STEP)) * int(STEP))  # resampled
  diffVal = y - yresampled
  cumError = cumError + diffVal

  if cumError > STEP:
      yresampled = yresampled + STEP
      cumError = cumError - STEP

  if cumError < (-1 * STEP):
      yresampled = yresampled - STEP
      cumError = cumError + STEP

  return yresampled


def iirfilter(x1):
  global v1m1, v2m1, v1m, v2m
  # print(v1m1, v2m1, v1m, v2m, cumError)
  y1 = 0
  y1 = (b[0] * x1 + v1m1) / a[0]
  v1m = (b[1] * x1 + v2m1) - a[1] * y1
  v2m = b[2] * x1 - a[2] * y1
  v1m1 = v1m
  v2m1 = v2m
  return y1
   
def band_pass_filter(signal):
  
  sig = signal.copy()

  for index in range(len(signal)):
    sig[index] = signal[index]

    if (index >= 1):
      sig[index] += 2*sig[index-1]

    if (index >= 2):
      sig[index] -= sig[index-2]

    if (index >= 6):
      sig[index] -= 2*signal[index-6]

    if (index >= 12):
      sig[index] += signal[index-12] 

  result = sig.copy()

  for index in range(len(signal)):
    result[index] = -1*sig[index]

    if (index >= 1):
      result[index] -= result[index-1]

    if (index >= 16):
      result[index] += 32*sig[index-16]

    if (index >= 32):
      result[index] += sig[index-32]

  max_val = max(max(result),-min(result))
  result = result*5/max_val

  return np.roll(result,-20)
   
def get_average_hearrate(arr):   
  differences = [abs(arr[i+1] - arr[i]) for i in range(len(arr)-1)]
  if len(differences) == 0:
    return 300
  avg_diff = sum(differences) / len(differences)
  return avg_diff

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

def accuracy(result, atr_index, tol):

  R_index = np.array([i for i, val in enumerate(result) if val == 1])

  tolerance = tol
  TP = sum(1 for r_value in R_index if any(abs(r_value - a_value) <= tolerance for a_value in atr_index))
  FP = len(R_index) - TP
  FN = len(atr_index) - TP

  return TP, FP, FN

def plot_result(x, hpass, lpass, result, atr_index, heights, lower, upper, tol, title):
  tolerance = tol
  actual_R = atr_index
  
  R_index = np.array([i for i, val in enumerate(result) if val == 1])
  TPs_idxs = [r_value for r_value in R_index if any(abs(r_value - a_value) <= tolerance for a_value in actual_R)]
  FPs_idxs = [r_value for r_value in R_index if not any(abs(r_value - a_value) <= tolerance for a_value in actual_R)]
  FNs_idxs = [a_value for a_value in actual_R if not any(abs(r_value - a_value) <= tolerance for r_value in R_index)]

  R_value_2 = np.array([lpass[i] for i, val in enumerate(result) if val == 2])
  R_index_2 = np.array([i for i, val in enumerate(result) if val == 2])

  TPs = [x[i] for i in TPs_idxs]
  FPs = [x[i] for i in FPs_idxs]
  FNs = [x[i] for i in FNs_idxs]
  
  filtered_r_val_real = [x[i] for i in atr_index]
  filtered_r_idx_real = atr_index

  TP = sum(1 for r_value in R_index if any(abs(r_value - a_value) <= tolerance for a_value in actual_R))
  FP = len(R_index) - TP
  FN = len(atr_index) - TP

  plt.plot(upper, color='red', alpha=0.4)
  plt.plot(lower, color='red', alpha=0.4)
  plt.plot(x)

  plt.scatter(TPs_idxs,TPs, color='orange')
  plt.scatter(FPs_idxs,FPs, color='red')
  plt.scatter(filtered_r_idx_real,filtered_r_val_real, color='yellow')
  plt.scatter(FNs_idxs,FNs, color='blue')
  plt.plot(lpass)
  plt.scatter(R_index_2,R_value_2, color='green')
  plt.plot(heights, color='lightgreen')
  plt.title(f"{title}  TP: {TP}  FP: {FP}  FN: {FN}", fontsize=12)
  plt.show()


def linear_HPF(signal, M):
  x = signal.copy()

  y2_hpass = np.roll(x, (int)((M+1)/2))
  y2_hpass[:(int)((M+1)/2)] = 0

  y1_hpass = x.copy()

  for i in range(1,M,1):
    y1_temp = np.roll(x, i)
    y1_temp[:i] = 0

    y1_hpass = y1_hpass + y1_temp

  y1_hpass = y1_hpass/M

  y_hpass = y2_hpass.copy()
  for i in range(len(y2_hpass)):
    y_hpass[i] = y2_hpass[i] - y1_hpass[i]

  y_hpass[:M-1] = 0
  return y_hpass


def nonlinear_LPF(signal, N):
  y_squared = signal.copy()

  for i in range(len(y_squared)):
    y_squared[i] = y_squared[i]**2

  y_lpass = y_squared.copy()

  for i in range(len(y_lpass)):
    low_limit = i - (N-1)
    if low_limit < 0:
      low_limit = 0
    y_lpass[i] = sum(y_squared[low_limit: i+1])

  return y_lpass

def quantize_0_10(signal, bits):
  clipped_array = np.clip(signal, 0, 10)
  scaled_array = np.interp(clipped_array, (0, 10), (0, (2**bits)-1))
  rounded_array = np.ceil(scaled_array).astype(np.int64)
  return rounded_array

def quantize(signal, bits):
  clipped_array = np.clip(signal, -5, 5)
  scaled_array = np.interp(clipped_array, (-5, 5), (0, (2**bits)-1))
  rounded_array = np.ceil(scaled_array).astype(np.int64)
  return rounded_array

def adaptive_thresholding_org(signal, winSize, gamma, alpha):
  threshold = 0
  thresholds = []
  triggered = False
  trig_time = 0
  win_max = 0
  win_idx = 0

  result = signal.copy()

  for i in range(len(signal)):

    point = signal[i]

    if i < winSize:
      if point > threshold:
        threshold = point

    if triggered:
      trig_time += 1
      if trig_time >= 100 :
          triggered = False
          trig_time = 0

    if(point > win_max):
      win_max = point

    if point > threshold and not triggered:
      result[i] = 1
      triggered = True
    else:
      result[i] = 0

    win_idx += 1
    if win_idx >= winSize:

      threshold = alpha * gamma * win_max + (1 - alpha) * threshold

      win_idx = 0
      win_max = 0

    thresholds.append(threshold)
  return result, thresholds

M = 5
N = 30
winSize = 250
R_count = 10
l_limit = 0
u_limit = 650000

train_DS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test_DS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# r = train_DS + test_DS
r = [100]
# r = [108]
# r = [108, 207]
# r = [108, 203, 207]
# steps = [11,10,9,8,7,6,5,3,2,1]
steps = [2]
# steps=[6]
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
    
    cumError = 0
    v1m1 = 0
    v2m1 = 0
    v1m = 0
    v2m = 0
    p_signal = p_signal * STEP
    print(original)
    original_1024 = (original/5) *(1<<10)
    # plt.plot(original_1024+6, label = 'Original')
    plt.plot(p_signal/(1<<10)*5, color = 'orange', label = 'Quantized')
    
    
    b_coeff, a_coeff = butter(3, 0.15, btype='low')
    p_signal = filtfilt(b_coeff, a_coeff, p_signal.copy())
    # plt.plot(p_signal, color = 'purple',label = 'Reconstructed')
    # plt.legend()
    # plt.show()
    p_signal = (p_signal/(1<<10)) * 5
    plt.plot(original, label = 'Original')
    plt.plot(p_signal, color = 'purple',label = 'Reconstructed')
    plt.legend()
    plt.show()
    # signal_min = -(1<<10)#min(p_signal)
    # signal_max = (1<<10)#max(p_signal)
    # scale = (10)/ (signal_max - signal_min)
    # shift = -5 - signal_min *scale # new_min - original_min * scale
    # p_signal = p_signal * scale + shift
    # plt.plot(p_signal, label = 'reconstructed')
    # plt.show()
    # ecg = np.array([iirfilter(i) for i in p_signal])

    # yresampled = np.array([pwm(i) for i in ecg])
    # plt.plot(ecg)
    # plt.plot(yresampled)
    # plt.show()
    # b_coeff, a_coeff = butter(3, 0.15, btype='low')
    # filtered_signal = filtfilt(b_coeff, a_coeff, yresampled.copy())
  
    # filtered_signal = filtered_signal.copy()/(1<<11)

    # ecg = quantize(sig, 8)
    # y_hpass = linear_HPF(ecg, M)
    # y_lpass = nonlinear_LPF(y_hpass, N)
    # result, thresholds, lower, upper = adaptive_thresholding_org(ecg, y_lpass, 0.175)
    # TP, FP, FN = accuracy(result, atr_sample, 30)
    # yresampled = np.array([pwm(i) for i in sig])
    # 100,0.02,0.1,0.3
    # tmp =  np.array([iirfilter(i) for i in p_signal])
    upper, lower = boundary(p_signal, 0.03)
    sig = np.array([iirfilter(i) for i in upper-lower])
    upper2, lower2 = boundary(sig , 0.035)
    # plt.plot(upper2)
    # plt.show()
    ecg = quantize(upper2 , 8)
    # y_hpass = np.zeros(len(upper2))
    # for i in range(len(upper2)-1):
    #   y_hpass[i] = upper2[i+1]-upper2[i]
    
    # sig = np.array([iirfilter(i) for i in ecg])
    y_hpass = linear_HPF(ecg, M)
    y_lpass = nonlinear_LPF(y_hpass, N)
    result, thersholds = adaptive_thresholding_org(y_lpass, 100, gamma= 0.25, alpha= 0.3)
    # result, thersholds = adaptive_thresholding_org(y_lpass, 250, gamma= 0.175, alpha= 0.05)
    TP, FP, FN = accuracy(result, atr_sample, 30)
    if writetofile:
      print([record, TP,FP,FN, round(TP*100/(TP+FN),2),round(TP*100/(TP+FN+FP),2),round(TP*100/(TP+FP),2)])

      ws.append([record, TP,FP,FN, round(TP*100/(TP+FN),2),round(TP*100/(TP+FN+FP),2),round(TP*100/(TP+FP),2)])
    else:
      print([record, TP,FP,FN, round(TP*100/(TP+FN),2),round(TP*100/(TP+FN+FP),2),round(TP*100/(TP+FP),2)])
    
      plot_result(p_signal, y_hpass, y_lpass, result, atr_sample, ecg, thersholds, [], 30, f"Record: {record} with steps {s}")
      # plt.plot(p_signal)
      # plt.plot(upper)
      # plt.plot(lower)
      # plt.plot(sig)
      # plt.plot(ecg, color='yellow')
      # plt.plot(upper2, color='red')
      # plt.plot(y_lpass)
      # plt.plot(lower2)
      # plt.plot(sig)
      # plt.plot(y_hpass)
      # plt.plot(upper-lower)
      # plt.plot(upper2, color='green')
      # plt.title(f"{TP},{FP},{FN}")
      # plt.show()
    # result = slidwind(p_signal)
    # plt.plot(p_signal)
    # plt.plot(result)
    # plt.show()
  
    # plt.plot(upper, color='red', alpha=0.4)
    # plt.plot(lower, color='green', alpha=0.4)
    print(record)

  if writetofile:
    ws.append(["", "","","", f"=average(e2:e{len(r)+1})",f"=average(f2:f{len(r)+1})",f"=average(g2:g{len(r)+1})"])
    wb.save(f"C:/Users/ISURI/Documents/fyp/compression/int_compressed_files/bits_vs_acc/bits_{s}.xlsx")
    # wb.save("Testing7/org_tol30_win250_gamma0.175_alpha0.05.xlsx")
  print("bits tested", s)