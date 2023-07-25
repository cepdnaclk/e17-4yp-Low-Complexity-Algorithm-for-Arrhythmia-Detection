# Python code transmits a byte to Arduino /Microcontroller

import serial

import time

SerialObj = serial.Serial('COM13') # COMxx   format on Windows
                                   # ttyUSBx format on Linux

SerialObj.baudrate = 115200  # set Baud rate to 9600
SerialObj.bytesize = 8     # Number of data bits = 8
SerialObj.parity   ='N'    # No parity
SerialObj.stopbits = 1     # Number of Stop bits = 1
SerialObj.timeout = 5


# Import necessary packages
import csv
  
# Open file
with open('p_signal_100.csv','r') as file_obj:
      
    # Create reader object by passing the file
    # object to DictReader method
    reader_obj = csv.reader(file_obj)
      
    # Iterate over each row in the csv file
    # using reader object
    for row in reader_obj:
        data = float(row[0])
        print(data)
        print("sleeping...")
        time.sleep(3)
        print("sending...")
        # bin = str(12.11)
        SerialObj.write(str(data).encode())      #transmit 'A' (8bit) to micro/Arduino

        print("waiting to receive...")
        time.sleep(3)
        # ReceivedString = SerialObj.readline()
        # print(ReceivedString)
        # SerialObj.close()        # Close the port

        record = SerialObj.readline()  # read null-terminated record
        # incomplete record read means timeout
        # if not record or record[-1] != 0:
        #     break
        # value = float(record.rstrip(b'\r\n\x00'))  # remove CR, LF and NUL at the end
        print(record)                               # do whatever with the value
        print("waiting to receive...")
        time.sleep(3)
        # ReceivedString = SerialObj.readline()
        # print(ReceivedString)
        # SerialObj.close()        # Close the port

        record = SerialObj.readline()  # read null-terminated record
        # incomplete record read means timeout
        # if not record or record[-1] != 0:
        #     break
        # value = float(record.rstrip(b'\r\n\x00'))  # remove CR, LF and NUL at the end
        print(record)                               # do whatever with the value
    SerialObj.close()  