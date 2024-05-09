#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POINTS 1000  // Number of points between 0 and 2*pi

// Define the annotation for TAFFO to use fixed-point precision
#define FIXED_POINT_PRECISION __attribute__((annotate("scalar(range(-1, 1) final)")))

// Floating-point sine calculation
void compute_sine_fp(float angles[], float sine_fp[], int size) {
    clock_t start = clock();
    for (int i = 0; i < size; i++) {
        sine_fp[i] = sin(angles[i]);
    }
    clock_t end = clock();
    printf("Floating-point computation time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}

// Fixed-point sine calculation with TAFFO annotation
void compute_sine_fx(float angles[], float sine_fx[], int size) {
    clock_t start = clock();
    for (int i = 0; i < size; i++) {
        float FIXED_POINT_PRECISION angle = angles[i];  // Correctly annotated variable for fixed-point operation
        sine_fx[i] = sin(angle);
    }
    clock_t end = clock();
    printf("Fixed-point computation time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}

// Function to calculate error
float calculate_error(float sine_fp[], float sine_fx[], int size) {
    float error = 0;
    for (int i = 0; i < size; i++) {
        error += fabs(sine_fp[i] - sine_fx[i]);
    }
    return error / size;  // Average error
}

int main() {
    float angles[NUM_POINTS];
    float sine_fp[NUM_POINTS];  // Floating-point results
    float sine_fx[NUM_POINTS];  // Fixed-point results

    // Initialize angle array
    for (int i = 0; i < NUM_POINTS; i++) {
        angles[i] = 2 * M_PI * i / NUM_POINTS;
    }

    // Compute sine values in floating-point
    compute_sine_fp(angles, sine_fp, NUM_POINTS);

    // Compute sine values in fixed-point
    compute_sine_fx(angles, sine_fx, NUM_POINTS);

    // Calculate error
    float error = calculate_error(sine_fp, sine_fx, NUM_POINTS);
    printf("Average error between fixed-point and floating-point: %f\n", error);

    return EXIT_SUCCESS;
}

