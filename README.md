___
# Developing Low Complexity Algorithms to Detect QRS Complex and Arrhythmia
___

## Introduction:


Arrhythmias or irregular heart rhythms are a significant component of cardiovascular diseases (CVD), a leading cause of global mortality. The electrocardiogram (ECG) is a vital tool widely used among healthcare professionals to diagnose and monitor these abnormalities. While the manual inspection of ECG signals in an acute condition is a difficult task, long-term monitoring of the electrical activity of the heart for early detection of transient or infrequent arrhythmias is also challenging without computer-aided diagnosis methods.

Computer-aided arrhythmia classification (CAAC) has become a well-researched topic with the development of public ECG databases. Over time, many sophisticated devices, from portable ECG monitoring devices to automated external defibrillators (AED), that can analyze the heart’s rhythm, identify the irregularities, and deliver an electrical shock to help the heart re-establish the normal rhythm if necessary have been developed with the help of CAAC. With the introduction of wearable ECG devices such as smartwatches, the possibility of real-time heart disease detection is now made available.

However, implementing efficient algorithms for arrhythmia detection from ECG signals in an environment with severe resource constraints remains challenging. In this paper, we propose a pre-packaged software solution containing a set of low-complexity algorithms to detect different arrhythmia conditions from ECG signals that can be implemented on microcontrollers with severely limited capabilities ($<$2kB SRAM, 1-8 MHz clock frequency).

To achieve this goal, we have introduced optimized solutions for the three main steps involved in state-of-the-art CAAC systems, namely, ECG Signal Denoising and Compression, QRS complex detection, and finally, the classifier.

Beat classification is the traditional approach for detecting arrhythmia. In this method, it is essential to accurately identify the QRS complexes in order to enable effective classification and analysis of ECG signals. However, achieving accurate QRS detection requires more complex methods that consume additional computational power and time. Consequently, this imposes an overhead on the actual arrhythmia detection process. Moreover, incorrect segmentation can negatively impact the overall classification of arrhythmia. To mitigate these unnecessary computations, our proposed method involves segmenting the ECG into fixed window sizes, each containing multiple beats. Then the class label of that segment is taken based on the highest severity class within that window.

The subsequent step involves converting ECG signals into sparse event-based representations. Delta modulation with adaptive thresholding is utilized to extract events from the QRS complexes. This event-based representation allows for a more concise data representation while reducing the computational load required for further processing and classification.

Finally, the classification is done on the encoded sparse, event-based ECG data using a two-stage Convolutional Spiking Neural Network (CSNN) trained using spike-timing dependent plasticity (STDP). The convolutional spiking neuron layers extract the features present within the input event stream.  The two-stage CSNN comprises a two-class CSNN to detect if the signal is normal. If it is normal, the classification is stopped, reducing the overhead of inference time and energy consumption. On the other hand, if the beat is not normal, the signal is further classified using a four-class CSNN. 

As this optimized solution can be implemented with minimal memory capacity and power consumption, it can be utilized to develop affordable, wearable devices for real-time cardiac health assessments for a broader population.
