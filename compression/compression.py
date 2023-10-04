import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import butter, filtfilt
import shutil


b = [0.0200833655642112, 0.0401667311284225, 0.0200833655642112]
a = [1, -1.56101807580072, 0.641351538057563]

# STEP = 1<<4

def load_ecg(filename):
    p_signal = []

    with open(f"CSVdata/p_signal_{filename}.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            p_signal.append(float(row[0]))

    p_signal = np.array(p_signal)


    return p_signal



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


def pwm(y):
  global cumError
#   y = y_*(1<<11)
  
  yresampled = ((int(y) // int(STEP)) * int(STEP))  # resampled
  diffVal = y - yresampled
  cumError = cumError + diffVal
  # print(f"y float {y}, yresampled {yresampled}, diff {diffVal}, cumError {cumError}, STEP, {STEP} ")
  if cumError > STEP:
      yresampled = yresampled + STEP
      cumError = cumError - STEP

  if cumError < (-1 * STEP):
      yresampled = yresampled - STEP
      cumError = cumError + STEP

  # if( yresampled/STEP <-3 or yresampled/STEP > 3):
  #     print("yresampled not in range", y, yresampled/STEP)
  # print(f"corrected yresampled {yresampled}")
  return yresampled

train_DS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test_DS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

r = train_DS + test_DS
steps = [11,10,9,8,7,6,5,4,3,2,1]
# steps=[8]

# r = [108]
for s in steps:
    STEP = 1 << s
    for record in r:
        cumError = 0
        v1m1 = 0
        v2m1 = 0
        v1m = 0
        v2m = 0

        p_signal = load_ecg(str(record))
        p_signal = (p_signal/5)* (1<<10)
        # plt.plot(p_signal, label = 'original')

        
        ecg = np.array([iirfilter(i) for i in p_signal])
        yresampled = np.array([pwm(i) for i in ecg])
        # print(ecg)
        # print(yresampled)
        # plt.plot(yresampled, label = 'resampled' )
        # plt.legend()
        # plt.show()

        b_coeff, a_coeff = butter(3, 0.15, btype='low')
        filtered_signal = filtfilt(b_coeff, a_coeff, yresampled.copy())
        # plt.plot(ecg, color = 'blue', label = 'original')
        # plt.plot(filtered_signal, color = 'orange', label = 'reconstructed')
        # plt.legend()
        # plt.show()


        ### Save to text file
        # Name of the output text file
        file_name = f"int_compressed_files/steps_{s}/compressed_{str(record)}.txt"
        yresampled_save = yresampled/STEP#[int(i/STEP) for i in yresampled]
        # yresampled_save = list(map(int, yresampled_save))
        yresampled_save = np.array(yresampled_save, dtype=int)
        np.savetxt(file_name, yresampled_save, fmt="%d", newline= '\n')
        print(record)
        # print(yresampled_save)
        # Open the file in write mode and create it if it doesn't exist
        # Open the file in write mode and create it if it doesn't exist
        # with open(file_name, "w") as file:
        #     # Use map and write to save list elements as rows in a single line
        #     map(file.write, yresampled_save)
        # plt.plot(yresampled_save*STEP)
        # plt.plot(yresampled, color = 'green')
        # plt.show()
    # shutil.make_archive(f"int_compressed_files/steps_{s}", 'zip', f"compressed_files/steps_{s}")
    print("steps completed ", s)