---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e17-4yp-Low-Complexity-Algorithm-for-Arrhythmia-Detection
title:
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Developing Low Complexity Algorithms to Detect QRS Complex and Arrhythmia from ECG signals
#### Team

- E/17/058, Devindi G.A.I, [e17058@eng.pdn.ac.lk](mailto:e17058@eng.pdn.ac.lk)
- E/17/190, Liyanage S.N , [e17190@eng.pdn.ac.lk](mailto:e17190@eng.pdn.ac.lk)


#### Supervisors

- Prof.Roshan G. Ragel, [roshanr@eng.pdn.ac.lk](mailto:roshanr@eng.pdn.ac.lk)
- Dr. Titus Jayarathna, [titus.Jayarathna@westernsydney.edu.au](mailto:titus.Jayarathna@westernsydney.edu.au)

#### Table of content

1. [Introduction](#introduction)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [References](#references)
10. [Links](#links)

<!--2. [Abstract](#abstract)-->
---
## Introduction

Arrhythmias or irregular heart rhythms are a significant component of cardiovascular diseases (CVD), a leading cause of global mortality. The electrocardiogram (ECG) is a vital tool widely used among healthcare professionals to diagnose and monitor these abnormalities. While the manual inspection of ECG signals in an acute condition is a difficult task, long-term monitoring of the electrical activity of the heart for early detection of transient or infrequent arrhythmias is also challenging without computer-aided diagnosis methods.

Computer-aided arrhythmia classification (CAAC) has become a well-researched topic with the development of public ECG databases. Over time, many sophisticated devices, from portable ECG monitoring devices to automated external defibrillators (AED), that can analyze the heart’s rhythm, identify the irregularities, and deliver an electrical shock to help the heart re-establish the normal rhythm if necessary have been developed with the help of CAAC. With the introduction of wearable ECG devices such as smartwatches, the possibility of real-time heart disease detection is now made available.

However, implementing efficient algorithms for arrhythmia detection from ECG signals in an environment with severe resource constraints remains challenging. In this paper, we propose a pre-packaged software solution containing a set of low-complexity algorithms to detect different arrhythmia conditions from ECG signals that can be implemented on microcontrollers with severely limited capabilities ($<$2kB SRAM, 1-8 MHz clock frequency).

To achieve this goal, we have introduced optimized solutions for the three main steps involved in state-of-the-art CAAC systems, namely, ECG Signal Denoising and Compression, QRS complex detection, and finally, the classifier.

Beat classification is the traditional approach for detecting arrhythmia. In this method, it is essential to accurately identify the QRS complexes in order to enable effective classification and analysis of ECG signals. However, achieving accurate QRS detection requires more complex methods that consume additional computational power and time. Consequently, this imposes an overhead on the actual arrhythmia detection process. Moreover, incorrect segmentation can negatively impact the overall classification of arrhythmia. To mitigate these unnecessary computations, our proposed method involves segmenting the ECG into fixed window sizes, each containing multiple beats. Then the class label of that segment is taken based on the highest severity class within that window.

The subsequent step involves converting ECG signals into sparse event-based representations. Delta modulation with adaptive thresholding is utilized to extract events from the QRS complexes. This event-based representation allows for a more concise data representation while reducing the computational load required for further processing and classification.

Finally, the classification is done on the encoded sparse, event-based ECG data using a two-stage Convolutional Spiking Neural Network (CSNN) trained using spike-timing dependent plasticity (STDP). The convolutional spiking neuron layers extract the features present within the input event stream.  The two-stage CSNN comprises a two-class CSNN to detect if the signal is normal. If it is normal, the classification is stopped, reducing the overhead of inference time and energy consumption. On the other hand, if the beat is not normal, the signal is further classified using a four-class CSNN. 

As this optimized solution can be implemented with minimal memory capacity and power consumption, it can be utilized to develop affordable, wearable devices for real-time cardiac health assessments for a broader population.

<!--## Abstract-->

## Related works
### ECG Signal Denoising and Compression
ECG signals are often affected by various types of noise, which can affect the accuracy of the diagnosis. The most significant noises are power line interference, muscular contractions, and baseline wander, which occur in different frequency ranges. ECG denoising techniques have evolved significantly over the years. Initially, classical filters like moving averages and FIR filters were used [1,2], but they have limitations in preserving waveform details. The advent of adaptive filters [3] allowed for dynamic noise estimation and removal. Wavelet transform-based methods provided [4,5] better time-frequency analysis, enabling noise suppression while preserving ECG features. These techniques have been developed and adapted to cater to low-power and real-time devices. 

Additionally, due to the large amount of data generated by continuous ECG monitoring, efficient compression techniques have evolved to efficiently store and transmit ECG data while minimizing the impact on diagnostic information. Transform-based methods like discrete cosine transform (DCT) and wavelet transform [6,7,8] were widely used. Wavelet-based compression balances compression ratio and signal quality, maintaining diagnostic features while reducing data size. Recent developments include using machine learning approaches, such as deep neural networks, for ECG compression, allowing for more advanced and adaptive compression algorithms [9, 10]. The evolution of ECG compression techniques continues to focus on achieving high compression ratios with minimal loss of important clinical information.

### QRS Complex Detection
The QRS complex is a prominent waveform in the ECG signal and plays a crucial role in automated heart rate determination and detecting cardiac abnormalities. In earlier days, template-based methods have been widely explored and involve the cross-correlation of ECG signals with predefined templates. [11, 12]. The template length and filter are determined experimentally. Although efficient implementations are available [13], the overall computational cost of QRS complex detection remains high due to the sample-by-sample moving comparison with the template across the ECG signals. Another widely explored area is derivative QRS detection methods [14]. Derivative-based methods are computationally efficient and can be implemented in real-time systems. However, these methods are sensitive to noise, and In the presence of abnormal or distorted ECG signals, derivative-based approaches may encounter challenges.

Filtering techniques have also been extensively investigated. Digital filters [15,16,17,18,19] The Pan-Tompkins algorithm [20] is a widely used and effective method for QRS complex detection. It employs low-pass, high-pass, and derivative filters to detect the QRS complex. While this algorithm demonstrates good performance, it may struggle with baseline wander and noise interference and may not perform optimally when faced with diverse morphologies. The Pan-Tompkins algorithm requires a relatively high computational load due to its iterative nature and multiple filtering stages. This can pose challenges for real-time processing applications with limited computational resources. Another widely used approach is wavelet-based QRS detection [21,22,23,24]. These methods are robust to noise. However, the wavelet transform involves complex calculations, and wavelet-based QRS detection methods can be computationally demanding. 

Neural networks are a widely used method in QRS detection. CNNs are commonly utilized for QRS detection tasks. They apply convolutional layers to learn local patterns and features from the ECG signal [25, 26]. The novel methods [27] provide noise-resistant and generalizable methods to detect QRS complexes accurately. These methods can be highly accurate. However, Neural networks, especially deep architectures, can be computationally demanding. Real-time QRS detection on low-power devices may face challenges due to the computational requirements of these methods. Optimizations, such as model compression or hardware acceleration, may be necessary. And these additional optimizations introduce additional overhead for actual ECG classification. 

Previous methods primarily focused on evaluating robustness to noise and numerical efficiency. However, they did not consider the performance based on time complexity or power consumption. Utilizing the first derivative of the filtered ECG signal, with or without a moving-average filter, is recommended as it offers high numerical efficiency during the QRS enhancement phase. However, this approach is sensitive to noise and arrhythmia, necessitating an adaptive thresholding or integration-based approach during the detection phase. Both of these methodologies are simple and computationally efficient, making them suitable for detecting QRS complexes in mobile-phone applications. Implementing the classical Pan-Tompkins algorithm is also a viable option in cases with more processing power available. Table I shows a summarized comparison of the different pre-processing steps taken for ECG classification.

![image](https://github.com/isuridevindi/e17-4yp-Low-Complexity-Algorithm-for-Arrhythmia-Detection/assets/71621792/ba0560c8-53f8-4cb3-8cde-1054c081019d)



### ECG Signal Classification Methods
Over the years, different techniques ranging from simple machine learning classifiers such as linear discriminants (LD) [28-30], decision trees [30-32], Naïve Bayes Classifiers [33], and k-Nearest Neighbors (k-NNs) [34] to more advanced models such as  Support Vector Machines (SVM) [35], feed-forward multi-layer perceptron (MLP) neural networks [36], hybrid neural networks [37], and a mixture of fuzzy logic, SVM, and neural networks [38,39] have been developed to classify ECG signals. These traditional implementations mainly have four steps; preprocessing the input ECG signals, extracting the features (attributes) from the preprocessed signals, selecting the subset of essential features from the extracted features, and feeding these features to the classifiers.

Out of the several publicly available databases containing annotated ECG signals [40-42], many previous works have utilized the MIT-BIH arrhythmia database consisting of 48 half-hours, two-channel ambulatory ECG recordings of 47 subjects (male: female 25:22; age range 23–89 years). Most literature has followed the standard from the Association for the Advancement of Medical Instrumentation (AAMI) and partitioned the MIT-BIH database into five groups: normal (N), supraventricular ectopic (SVE), ventricular ectopic (VE), fusion (F), and unknown (Q).
Even though it is necessary to increase the number of classes for a more qualitative analysis of the ECG signals, most work report results for either two [43], four [44,45], or five [46] class classifications to compensate for higher performance. This is in part because the feature extraction stage of traditional methods discards critical information.

Deep neural networks have been introduced in recent years to overcome the issues in traditional approaches. For example, in [47], a group of researchers implemented a shallow 11-layer convolutional neural network (CNN) to classify four classes of ECG, eliminating the need for different feature extraction techniques. Furthermore, in [48], a 34-layer CNN has been developed to classify an arbitrary length ECG time series into 14 arrhythmia classes from single-lead ECG signals better than a cardiologist. Much other work that utilizes a combination of deep neural networks, such as CNN, RNN, and LSTMs, has been introduced over the years. For instance, in [49], the authors present a CNN and a bidirectional long-term, short-term memory network (BiLSTM) model to automatically classify ECG heartbeats into five groups in the AAMI standard. Even though these works demonstrate high accuracy, these deep networks dramatically increase computation costs, power consumption, deployment complexity, and inference time.
As a low-power, low-computation cost alternative, the third generation of neural networks, Spiking Neural Networks (SNN), have been explored in the most recent works. As SNNs use the Leaky Integrate and Fire (LIF) neuron models, which transmit information via non-differentiable discrete spikes, standard backpropagation methods are impractical for training SNNs. As a workaround, in [50], a two-stage CNN workflow is proposed for training, with early stopping to reduce the time, while the trained CNN models are converted to SNNs to minimize power consumption at inference time. On the other hand, [51] introduces a binarised SNN that converts high precision weights of the network to binary values, reducing the computational complexity further, while using the spike-time dependent plasticity  (STDP) learning algorithm to directly train an SNN without backpropagation. Most previous SNN-based work employs non-event-based algorithms [52,53] or simple rate encoding [50]  for ECG signal pattern extraction and encoding. Therefore, [54] proposes an SNN-based pattern extraction method using an STDP layer with the help of a Gaussian layer and an inhibitory layer. Table II shows a summarized comparison of the classification methods categorized into three main types.

Even though SNN-based methods show a significant improvement in energy consumption, these implementations are still incapable of classifying ECG signals in a microcontroller with severely limited capabilities ($<$2kB SRAM, 1-8 MHz clock frequency). On the other hand, converting and compressing the ECG signal to a sparse, event-based representation, without loss of information and without affecting the classification accuracy for transmission remains an unaddressed challenge.

![image](https://github.com/isuridevindi/e17-4yp-Low-Complexity-Algorithm-for-Arrhythmia-Detection/assets/71621792/be9169cb-4348-4966-98a8-b80f8761252d)


## Methodology

## Experiment Setup and Implementation

## Results and Analysis

## Conclusion

## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->

## References
1] V. Pandey and V. K. Giri, ”High-frequency noise removal from ECG
using moving average filters,” 2016 International Conference on Emerg-
ing Trends in Electrical Electronics & Sustainable Energy Systems
(ICETEESES), Sultanpur, India, 2016, pp. 191-195, doi: 10.1109/ICE-
TEESES.2016.7581383.

[2] Sonali, O. Singh and R. K. Sunkaria, ”ECG signal denoising based
on Empirical Mode Decomposition and moving average filter,” 2013
IEEE International Conference on Signal Processing, Computing and
Control (ISPCC), Solan, India, 2013, pp. 1-6, doi: 10.1109/IS-
PCC.2013.6663412.

[3] An, X., Liu, Y., Zhao, Y., Lu, S., Stylios, G. K., & Liu, Q.
(2022). Adaptive Motion Artifact Reduction in Wearable ECG Measure-
ments Using Impedance Pneumography Signal. Sensors, 22(15), 5493.
https://doi.org/10.3390/s22155493

[4] C. -Y. Wang, D. -J. Huang, X. -Q. Xie, Y. -P. Zhang and L. -H. Wang, ”A
lifting double-wavelet algorithm for ECG signal denoising,” 2016 IEEE
International Conference on Consumer Electronics-Taiwan (ICCE-TW),
Nantou, Taiwan, 2016, pp. 1-2, doi: 10.1109/ICCE-TW.2016.7520920.

[5] S. Hadji, M. Salleh, M. Rohani, and M. Kamat. 2016. Wavelet-based
Performance in Denoising ECG Signal. In Proceedings of the 8th
International Conference on Signal Processing Systems (ICSPS 2016).
Association for Computing Machinery, New York, NY, USA, 148–153.
https://doi.org/10.1145/3015166.3015212

[6] B. A. Rajoub, ”An efficient coding algorithm for the compression of
ECG signals using the wavelet transform,” in IEEE Transactions on
Biomedical Engineering, vol. 49, no. 4, pp. 355-362, April 2002, doi:
10.1109/10.991163.

[7] A. Djohan, T. Q. Nguyen and W. J. Tompkins, ”ECG compression
using discrete symmetric wavelet transform,” Proceedings of 17th
International Conference of the Engineering in Medicine and Biol-
ogy Society, Montreal, QC, Canada, 1995, pp. 167-168 vol.1, doi:
10.1109/IEMBS.1995.575053.

[8] J. Chen and S. Itoh, ”A wavelet transform-based ECG compression
method guaranteeing desired signal quality,” in IEEE Transactions on
Biomedical Engineering, vol. 45, no. 12, pp. 1414-1419, Dec. 1998,
doi: 10.1109/10.730435.

[9] Hua J, Chu B, Zou J, Jia J ”ECG signal classification in wearable
devices based on compressed domain”. PLOS ONE 18(4): e0284008.
https://doi.org/10.1371/journal.pone.0284008

[10] Lijuan Zheng and Zihan Wang and Junqiang Liang and Shifan Luo
and Senping Tian, ”Effective compression and classification of ECG
arrhythmia by singular value decomposition”, vol. 2, p. 100013 doi:
10.1016/j.bea.2021.100013

[11] Kaplan D (1990) Simultaneous QRS detection and feature extraction
using simple matched filter basis functions. In: Proc. IEEE Computers
in Cardiology. pp. 503–506

[12] Ruha A, Sallinen S, Nissila S (1997) A real-time microprocessor QRS
detector system with a 1-ms timing accuracy for the measurement of
ambulatory HRV. IEEE Trans Biomedical Engineering 44: 159–167.

[13] Eskofier B, Kornhuber J, Hornegger J (2008) Embedded QRS detection
for noisy ECG sensor data using a matched filter and directed graph
search. In: Proc. 4th Russian-Bavarian Conference on Biomedical En-
gineering, Zelenograd, Moscow, Russia, pp. 48–52.

[14] M. Adnane, Z. Jiang, S. Choi, ”Development of QRS detection algorithm
designed for wearable cardiorespiratory system”, Computer Methods and
Programs in Biomedicine, Volume 93, Issue 1, 2009, Pages 20-31, ISSN
0169-2607, doi: 10.1016/j.cmpb.2008.07.010.

[15] L. D. Sharma, R. K. Sunkaria, ”A robust QRS detection using novel
pre-processing techniques and kurtosis based enhanced efficiency”,
Measurement, Volume 87, 2016, Pages 194-204, ISSN 0263-2241, doi:
10.1016/j.measurement.2016.03.015.

[16] D. Castells-Rufas, J. Carrabina, ”Simple real-time QRS detector
with the MaMeMi filter”, Biomedical Signal Processing and Con-
trol, Volume 21, 2015, Pages 137-145, ISSN 1746-8094, doi:
10.1016/j.bspc.2015.06.001.

[17] A. K. Dohare, V. K., R. Kumar, ”An efficient new method for the
detection of QRS in electrocardiogram”, Computers & Electrical Engi-
neering, Volume 40, Issue 5, 2014, Pages 1717-1730, ISSN 0045-7906,
doi: 10.1016/j.compeleceng.2013.11.004.

[18] U. Pangerc and F. Jager, ”Robust detection of heart beats in multimodal
records using slope- and peak-sensitive band-pass filters”, Physiological
Measurement, vol. 36, no. 8, p. 1645, Jul. 2015. doi: 10.1088/0967-
3334/36/8/1645

[19] Mourad, K., & Fethi, B. R. (2016). ”Efficient automatic detection of
QRS complexes in ECG signal based on reverse biorthogonal wavelet
decomposition and nonlinear filtering”. Measurement, 94, 663-670. doi:
10.1016/j.measurement.2016.09.014

[20] J. Pan and W. J. Tompkins, ”A Real-Time QRS Detection Algorithm,”
in IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3,
pp. 230-236, March 1985, doi: 10.1109/TBME.1985.325532.

[21] Yochum, M., Renaud, C., & Jacquir, S. (2016). ”Automatic detec-
tion of P, QRS and T patterns in 12 leads ECG signal based on
CWT”. Biomedical Signal Processing and Control, 25, 46-52. doi:
10.1016/j.bspc.2015.10.011

[22] Di Marco, L.Y., Chiari, L. ”A wavelet-based ECG delineation algorithm
for 32-bit integer online processing”. BioMed Eng OnLine 10, 23 (2011).
doi: 10.1186/1475-925X-10-23

[23] Ghaffari, A., Homaeinezhad, M.R., Khazraee, M. et al. ”Segmentation of
Holter ECG Waves Via Analysis of a Discrete Wavelet-Derived Multiple
Skewness–Kurtosis Based Metric”. Ann Biomed Eng 38, 1497–1510
(2010). doi: 10.1007/s10439-010-9919-3

[24] Mourad, K., & Fethi, B. R. (2016). ”Efficient automatic detection of
QRS complexes in ECG signal based on reverse biorthogonal wavelet
decomposition and nonlinear filtering”. Measurement, 94, 663-670. doi:
10.1016/j.measurement.2016.09.014

[25] M. ˇSarlija, F. Juriˇsi ́c and S. Popovi ́c, ”A convolutional neural network
based approach to QRS detection,” Proceedings of the 10th International
Symposium on Image and Signal Processing and Analysis, Ljubljana,
Slovenia, 2017, pp. 121-125, doi: 10.1109/ISPA.2017.8073581.

[26] Belkadi, M. A., Daamouche, A., & Melgani, F. (2021). ”A deep neural
network approach to QRS detection using autoencoders”. Expert Sys-
tems with Applications, 184, 115528. doi: 10.1016/j.eswa.2021.115528

[27] W. Cai and D. Hu, ”QRS Complex Detection Using Novel Deep
Learning Neural Networks,” in IEEE Access, vol. 8, pp. 97082-97089,
2020, doi: 10.1109/ACCESS.2020.2997473.

[28] P. deChazal, M. O’Dwyer, and R. B. Reilly, “Automatic Classification
of Heartbeats Using ECG Morphology and Heartbeat Interval Features,”
IEEE Transactions on Biomedical Engineering, vol. 51, no. 7, pp.
1196–1206, Jul. 2004, doi: https://doi.org/10.1109/tbme.2004.827359.

[29] T. Mar, S. Zaunseder, J. P. Martinez, M. Llamedo, and R. Poll, “Opti-
mization of ECG Classification by Means of Feature Selection,” IEEE
Transactions on Biomedical Engineering, vol. 58, no. 8, pp. 2168–2177,
Aug. 2011, doi: https://doi.org/10.1109/tbme.2011.2113395.

[30] V. Krasteva, I. Jekova, R. Leber, R. Schmid, and R. Ab ̈acherli, “Superior-
ity of Classification Tree versus Cluster, Fuzzy and Discriminant Models
in a Heartbeat Classification System,” PLOS ONE, vol. 10, no. 10, p.
e0140123, Oct. 2015, doi: https://doi.org/10.1371/journal.pone.0140123.

[31] S. Sultan Qurraie and R. Ghorbani Afkhami, “ECG arrhythmia clas-
sification using time frequency distribution techniques,” Biomedical
Engineering Letters, vol. 7, no. 4, pp. 325–332, Jul. 2017, doi:
https://doi.org/10.1007/s13534-017-0043-2.

[32] R. Ghorbani Afkhami, G. Azarnia, and M. A. Tinati, “Cardiac arrhyth-
mia classification using statistical and mixture modeling features of ECG
signals,” Pattern Recognition Letters, vol. 70, pp. 45–51, Jan. 2016, doi:
https://doi.org/10.1016/j.patrec.2015.11.018.

[33] D. Gao, M. Madden, D. C. Chambers, and G. Lyons, “Bayesian ANN
classifier for ECG arrhythmia diagnostic system: a comparison study,”
Jan. 2006, doi: https://doi.org/10.1109/ijcnn.2005.1556275.

[34] Saeed Karimifard, A. Ahmadian, Mohammad Hossein Khoshnevisan,
and Mohammad Saleh Nambakhsh, “Morphological Heart Arrhythmia
Detection Using Hermitian Basis Functions and kNN Classifier,” Aug.
2006, doi: https://doi.org/10.1109/iembs.2006.260182.

[35] S. Raj, K. C. Ray, and O. Shankar, “Cardiac arrhythmia beat classi-
fication using DOST and PSO tuned SVM,” Computer Methods and
Programs in Biomedicine, vol. 136, pp. 163–177, Nov. 2016, doi:
https://doi.org/10.1016/j.cmpb.2016.08.016.

[36] O. T. Inan, L. Giovangrandi, and G. T. A. Kovacs, “Robust Neural-
Network-Based Classification of Premature Ventricular Contractions
Using Wavelet Transform and Timing Interval Features,” IEEE Transac-
tions on Biomedical Engineering, vol. 53, no. 12, pp. 2507–2515, Dec.
2006, doi: https://doi.org/10.1109/tbme.2006.880879.

[37] Z. Dokur and T.  ̈Olmez, “ECG beat classification by a novel hybrid
neural network,” Computer Methods and Programs in Biomedicine, vol.
66, no. 2–3, pp. 167–181, Sep. 2001, doi: https://doi.org/10.1016/s0169-
2607(00)00133-4.

[38] S. Osowski and Tran Hoai Linh, “ECG beat recognition
using fuzzy hybrid neural network,” IEEE Transactions on
Biomedical Engineering, vol. 48, no. 11, pp. 1265–1271, 2001,
doi:https://doi.org/10.1109/10.959322.

[39] N. O. Ozcan and F. Gurgen, “Fuzzy Support Vector Machines for ECG
Arrhythmia Detection,” 2010 20th International Conference on Pattern
Recognition, Aug. 2010, doi: https://doi.org/10.1109/icpr.2010.728.

[40] G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia
Database,” IEEE Engineering in Medicine and Biology Magazine, vol.
20, no. 3, pp. 45–50, 2001, doi: https://doi.org/10.1109/51.932724.

[41] Nolle FM, Badura FK, Catlett JM, Bowser RW, Sketch MH. CREI-
GARD, a new concept in computerized arrhythmia monitoring systems.
Computers in Cardiology 13:515-518 (1986).

[42] Goldberger A, Amaral L, Glass L, Hausdorff J, Ivanov PC, Mark R, Mi-
etus JE, Moody GB, Peng CK, Stanley HE. PhysioBank, PhysioToolkit,
and PhysioNet: Components of a new research resource for complex
physiologic signals. Circulation [Online]. 101 (23), pp. E215–e220.

[43] P. Cheng and X. Dong, “Life-Threatening Ventricular Arrhythmia
Detection With Personalized Features,” IEEE Access, vol. 5, pp.
14195–14203, 2017, doi: https://doi.org/10.1109/access.2017.2723258.

[44] Yu Hen Hu, S. Palreddy, and W. J. Tompkins, “A patient-adaptable ECG
beat classifier using a mixture of experts approach,” IEEE Transactions
on Biomedical Engineering, vol. 44, no. 9, pp. 891–900, 1997, doi:
https://doi.org/10.1109/10.623058.

[45] S. Palreddy, W. J. Tompkins, and Y. Hu, “Customization of ECG
beat classifiers developed using SOM and LVQ,” Sep. 1995, doi:
https://doi.org/10.1109/iembs.1995.575376.

[46] R. Silipo, M. Gori, A. Taddei, Maurizio Varanini, and C. Marchesi,
“Classification of Arrhythmic Events in Ambulatory Electrocardiogram,
Using Artificial Neural Networks,” vol. 28, no. 4, pp. 305–318, Aug.
1995, doi: https://doi.org/10.1006/cbmr.1995.1021.

[47] U. R. Acharya, H. Fujita, S. L. Oh, Y. Hagiwara, J. H. Tan, and
M. Adam, “Application of deep convolutional neural network for
automated detection of myocardial infarction using ECG signals,”
Information Sciences, vol. 415–416, pp. 190–198, Nov. 2017, doi:
https://doi.org/10.1016/j.ins.2017.06.027.

[48] P. Rajpurkar, A. Hannun, M. Haghpanahi, C. Bourn, and A. Ng,
“Cardiologist-Level Arrhythmia Detection with Convolutional Neural
Networks.” Available: https://arxiv.org/pdf/1707.01836.pdf

[49] S. Bhatia, S. K. Pandey, A. Kumar, and A. Alshuhail, “Classifica-
tion of Electrocardiogram Signals Based on Hybrid Deep Learning
Models,” Sustainability, vol. 14, no. 24, p. 16572, Dec. 2022, doi:
https://doi.org/10.3390/su142416572.

[50] Z. Yan, J. Zhou, and W.-F. Wong, “Energy efficient ECG
classification with spiking neural network,” Biomedical Signal
Processing and Control, vol. 63, p. 102170, Jan. 2021, doi:
https://doi.org/10.1016/j.bspc.2020.102170.

[51] A. Rana and K. K. Kim, “A Novel Spiking Neural Network
for ECG signal Classification,” JOURNAL OF SENSOR SCIENCE
AND TECHNOLOGY, vol. 30, no. 1, pp. 20–24, Jan. 2021, doi:
https://doi.org/10.46670/jsst.2021.30.1.20.

[52] E. Kolagasioglu and A. Zjajo, “Energy Efficient Feature Extraction
for Single-Lead ECG Classification Based On Spiking Neural Net-
works,”Ph.D. dissertation, Delft University of Technology, 2018.

[53] Y. Xing et al., “Accurate ECG Classification Based on Spiking Neural
Network and Attentional Mechanism for Real-Time Implementation on
Personal Portable Devices,” Electronics, vol. 11, no. 12, p. 1889, Jan.
2022, doi: https://doi.org/10.3390/electronics11121889.

[54] A. Amirshahi and M. Hashemi, “ECG Classification Algorithm Based
on STDP and R-STDP Neural Networks for Real-Time Monitoring on
Ultra-Low-Power Personal Wearable Devices,” IEEE Transactions on
Biomedical Circuits and Systems, vol. 13, no. 6, pp. 1483–1493, Dec.
2019, doi: https://doi.org/10.1109/tbcas.2019.2948920

## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository]((https://github.com/cepdnaclk/e17-4yp-Low-Complexity-Algorithm-for-Arrhythmia-Detection/))
- [Project Page](https://cepdnaclk.github.io/e17-4yp-Low-Complexity-Algorithm-for-Arrhythmia-Detection/)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
