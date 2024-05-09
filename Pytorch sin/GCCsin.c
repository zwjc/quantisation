#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POINTS 1000  // Number of points between 0 and 2*pi

// Function to compute sine values using standard floating-point arithmetic
void compute_sine_fp(float angles[], float sine_values[], int size) {
    for (int i = 0; i < size; i++) {
        sine_values[i] = sin(angles[i]);
    }
}

// Function to print the computed sine values (optional, for verification)
void print_sine_values(float sine_values[], int size) {
    for (int i = 0; i < size; i++) {
        printf("sin(%f) = %f\n", (2 * M_PI * i / size), sine_values[i]);
    }
}

int main() {
    float angles[NUM_POINTS];
    float sine_values[NUM_POINTS];

    // Initialize angle array
    for (int i = 0; i < NUM_POINTS; i++) {
        angles[i] = 2 * M_PI * i / NUM_POINTS;
    }

    // Compute sine values in floating-point
    compute_sine_fp(angles, sine_values, NUM_POINTS);

    // Optionally print sine values
    // print_sine_values(sine_values, NUM_POINTS);

    return EXIT_SUCCESS;
}
