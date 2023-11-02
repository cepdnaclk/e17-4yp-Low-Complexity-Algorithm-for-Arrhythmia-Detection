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


int main() {
    int signal_length = 100;
    int signal [] = {995,
995,
995,
995,
995,
995,
995,
995,
1000,
997,
995,
994,
992,
993,
992,
989,
988,
987,
990,
993,
989,
988,
986,
988,
993,
997,
993,
986,
983,
977,
979,
975,
974,
972,
969,
969,
969,
971,
973,
971,
969,
966,
966,
966,
966,
967,
965,
963,
967,
969,
969,
968,
967,
963,
966,
964,
968,
966,
964,
961,
960,
957,
952,
947,
947,
943,
933,
927,
927,
939,
958,
980,
1010,
1048,
1099,
1148,
1180,
1192,
1177,
1128,
1058,
991,
951,
937,
939,
950,
958,
959,
957,
955,
958,
959,
961,
962,
960,
957,
956,
959,
955,
957};  // Replace with actual data

    // Call the 'reconstruct' function
    int bit = 4;
    int *compressed = compressEcgSignal(signal, 100, 11, 10, 360, 180);

    for(int i=0; i< signal_length; i++){
        printf("%d ", compressed[i]);
    }
    printf("\n");
    // Free memory allocated for the filtered signal
    free(compressed);

    return 0;
}
