// macos version
#include "LBP.h"
#include <math.h>

using namespace std;

// S = 1/M * sum || h_i^en - h_i^ver ||   (M = 49, 작을수록 같은 사람)
float computeSimilarity(const vector<float>& ref, const vector<float>& tar)
{
    int M = NBLOCK * NBLOCK;
    float total = 0;

    for (int i = 0; i < M; i++) {
        float d = 0;
        for (int k = 0; k < NBIN; k++) {
            float diff = ref[i * NBIN + k] - tar[i * NBIN + k];
            d += diff * diff;
        }
        total += sqrt(d);
    }
    return total / M;
}