#include <stdio.h>
#include <stdlib.h>

// Global variable declaration
int cumError = 0;

// pwm 
int compressEcgPoint(int y, int org_bit, int down_bit) {

    if (org_bit <= down_bit) return y;

    // Resample 
    int STEP = 1 << (org_bit-down_bit);
    int yresampled = ((int)(y / STEP)) * STEP;
    int diffVal = y - yresampled;

    // Update cumulative error
    cumError = cumError + diffVal;

    if ((cumError > STEP) && (yresampled + STEP < (1<<org_bit))) {
        yresampled = yresampled + STEP;
        cumError = cumError - STEP;
    }

    if (cumError < (-1 * STEP)) {
        yresampled = yresampled - STEP;
        cumError = cumError + STEP;
    }

    return (int)(yresampled / STEP);
}


int* compressEcgSignal(int* p_signal, int signal_len, int org_bit, int down_bit, int org_fs, int down_fs) {
    cumError = 0;

    if (org_fs < down_fs) return p_signal;

    // Calculate the size of the downsampled signal array
    int downsampling_factor = org_fs / down_fs;
    int downsampled_size = signal_len / downsampling_factor;

    // Allocate memory for the downsampled signal
    int* result = (int*)malloc(downsampled_size * sizeof(int));

    // Downsample the signal
    for (int i = 0; i < downsampled_size; i++) {
        result[i] = compressEcgPoint(p_signal[i * downsampling_factor], org_bit, down_bit);
    }

    return result;
}
