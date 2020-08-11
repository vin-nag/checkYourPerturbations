#include <math.h>
#include <stdio.h>
int main() {
float feature_0 = 1.0;// __VERIFIER_assume(feature_0 <= 1.0f && feature_0 >= -1.0f && fabsf(feature_0 - 1.0) < 0.01f);
float feature_1 = 1.0;// __VERIFIER_assume(feature_1 <= 1.0f && feature_1 >= -1.0f && fabsf(feature_1 - 1.0) < 0.01f);
float neuron0_0 = 0.36775755882263184f * feature_0 +-0.12151980400085449f * feature_1 +0.0;
float layer0_0 = (neuron0_0 + fabsf(neuron0_0)) / 2.0f;
float neuron0_1 = 0.7610447406768799f * feature_0 +0.01985025405883789f * feature_1 +0.0;
float layer0_1 = (neuron0_1 + fabsf(neuron0_1)) / 2.0f;
float neuron0_2 = 0.2110004425048828f * feature_0 +-0.7795701026916504f * feature_1 +0.0;
float layer0_2 = (neuron0_2 + fabsf(neuron0_2)) / 2.0f;
float neuron0_3 = -0.939673662185669f * feature_0 +0.5561661720275879f * feature_1 +0.0;
float layer0_3 = (neuron0_3 + fabsf(neuron0_3)) / 2.0f;
float output_0 = -0.9027526378631592*layer0_0 + 0.6788184642791748*layer0_1 + -0.17794084548950195*layer0_2 + 0.875464677810669*layer0_3 + 0.0;
float output_1 = -0.5832931995391846*layer0_0 + 0.23112249374389648*layer0_1 + 0.4602816104888916*layer0_2 + -0.47504305839538574*layer0_3 + 0.0;
float sum = expf(output_0) + expf(output_1);
printf("%f %f\n", output_0, output_1);
printf("%f %f\n", expf(output_0)/sum, expf(output_1)/sum);
// assert(output_1 > output_0 );
return 0;
}