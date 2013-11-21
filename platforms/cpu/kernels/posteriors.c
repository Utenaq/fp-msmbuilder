#include "math.h"
#include "posteriors.h"
#include "logsumexp.h"

void compute_posteriors(const float* __restrict__ fwdlattice,
                        const float* __restrict__ bwdlattice,
                        const int sequence_length,
                        const int n_states,
                        float* __restrict__ posteriors)
{
    int t, i;
    float gamma[n_states];
    float normalizer;

    for (t = 0; t < sequence_length; t++) {
        for (i = 0; i < n_states; i++)
            gamma[i] = fwdlattice[t*n_states + i] + bwdlattice[t*n_states + i];
        normalizer = logsumexp(gamma, n_states);
        for (i = 0; i < n_states; i++)
            posteriors[t*n_states+i] = exp(gamma[i] - normalizer);
    }
}

