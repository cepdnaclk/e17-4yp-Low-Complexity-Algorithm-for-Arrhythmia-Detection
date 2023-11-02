import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import butter, filtfilt
import argparse

####
# The input file should be a decimal txt file
# step_size: take the number of bits to be reduced as an argument [1,2,3,4,5,6,7,8,9,10]
# sampling_rate: take the downsampling rate as an arguments [360,180,120]
# reconstruct: reconstruct the compressed signal
# visualize: visualize results

####

b = [0.0200833655642112, 0.0401667311284225, 0.0200833655642112]
a = [1, -1.56101807580072, 0.641351538057563]


def load_ecg(filename):
    p_signal = []

    with open(filename, 'r') as file:

        csv_reader = csv.reader(file)
        for row in csv_reader:
            p_signal.append(float(row[0]))

    p_signal = np.array(p_signal)


    return p_signal

def pwm(y,STEP):
  global cumError
  yresampled = ((int(y) // int(STEP)) * int(STEP))  # resampled
  diffVal = y - yresampled
  cumError = cumError + diffVal

  if cumError > STEP:
      yresampled = yresampled + STEP
      cumError = cumError - STEP

  if cumError < (-1 * STEP):
      yresampled = yresampled - STEP
      cumError = cumError + STEP

  return yresampled


def main():

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Input digital ECG data txt file, step size and downsampling rate")

    # Add arguments for file path, sampling rate, and step size
    parser.add_argument("file_path", type=str, help="Path to the text file to process")
    parser.add_argument("--sampling_rate", type=int, default=360, help="Sampling rate (default: 360)")
    parser.add_argument("--step_size", type=int, default=4, help="Step size (default: 4)")
    parser.add_argument("--reconstruct",action = 'store_true', help="Enable to reconstruct the signal")
    parser.add_argument("--visualize",action = 'store_true', help="Enable to visualize the signal")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the arguments
    record = args.file_path
    downsampled_sampling_rate = args.sampling_rate
    s = args.step_size
    reconstruct = args.reconstruct
    visualize = args.visualize

    
    print(f"File Path: {record}")
    print(f"Sampling Rate: {downsampled_sampling_rate}")
    print(f"Step Size: {s}")
    print(f"Reconstruct: {reconstruct}")
    print(f"Visualize: {visualize}")

    STEP = 1<<(11-s)
    p_signal = load_ecg(str(record))
    original = p_signal

    # Calculate the time range for the original data
    original_sampling_rate = 360  # Hz
    original_duration = len(p_signal) / original_sampling_rate
    original_time = np.linspace(0, original_duration, len(p_signal))
    downsampled_duration = original_duration
    downsampled_time = original_time

    if(downsampled_sampling_rate !=360):
    
        downsampling_factor =int(original_sampling_rate/downsampled_sampling_rate)# 2
        p_signal = p_signal[::downsampling_factor]
        downsampled_duration = len(p_signal) / downsampled_sampling_rate
        downsampled_time = np.linspace(0, downsampled_duration, len(p_signal))

    if (visualize):
        # Plot both the original and downsampled data
        plt.figure(figsize=(10, 6))
        plt.plot(original_time, original, label='Original Data', color='b')
        
        plt.plot(downsampled_time, p_signal, label='Downsampled Data', color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Original vs Downsampled Data')
        plt.grid(True)
        plt.show()

    yresampled = np.array([pwm(i,STEP) for i in p_signal])

    ### Save to text file
    # Name of the output text file
    file_name = f"compressed_signal_{s}bits_{downsampled_sampling_rate}Hz.txt"
    yresampled_save = yresampled/STEP
    yresampled_save = np.array(yresampled_save, dtype=int)
    np.savetxt(file_name, yresampled_save, fmt="%d", newline= '\n')
    print("File saved: ",record)


    if (reconstruct):
        b_coeff, a_coeff = butter(3, 0.15, btype='low')
        filtered_signal = filtfilt(b_coeff, a_coeff, yresampled.copy())
        p_signal = filtfilt(b_coeff, a_coeff, p_signal)
        upsampled_data = filtered_signal
        upsampled_time = original_time

        if(downsampled_sampling_rate !=360):
        # Upsample the data to 360Hz using linear interpolation
            upsampled_sampling_rate = 360  # Hz
            upsampled_duration = downsampled_duration
            upsampled_time = np.linspace(0, upsampled_duration, len(filtered_signal) * downsampling_factor-1)  # Double the number of points for 360Hz

            # Use linear interpolation to upsample the data
            upsampled_data = np.interp(upsampled_time, downsampled_time, filtered_signal)
        
        if(visualize):

            plt.plot(original_time,original, color = 'blue', label = 'original')
            plt.plot(upsampled_time,upsampled_data, color = 'orange', label = 'reconstructed')
            plt.legend()
            plt.show()
  
if __name__ == "__main__":
    
 
    cumError = 0
    v1m1 = 0
    v2m1 = 0
    v1m = 0
    v2m = 0
        
    main()

    