/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <benchmark/benchmark.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"


// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size


int enzyme_dup;
int enzyme_const;
// for gradient checking
double EPSILON = 1e-4;
double TOLERANCE = 1e-3;
float __enzyme_autodiff(void *, ...);

// finite distance checking for f: R^n -> R
void check_gradients(
    float *numerical_gradients,   // Perturbed values (numerical approximation)
    float *analytical_gradients, // Gradients from backpropagation
    int size,                  // Size of the vector
    float epsilon,             // Perturbation value
    float tolerance            // Acceptable tolerance for error
) {
    for (int i = 0; i < size; i++) {
        // Compute numerical gradient using finite differences
        float numerical_gradient = numerical_gradients[i];

        // Compare numerical gradient with analytical gradient
        float difference = fabs(numerical_gradient - analytical_gradients[i]);
        if (difference > tolerance) {
            printf("Gradient check failed at index %d:\n", i);
            printf("  Analytical Gradient: %f\n", analytical_gradients[i]);
            printf("  Numerical Gradient: %f\n", numerical_gradient);
            printf("  Difference: %f (tolerance: %f)\n", difference, tolerance);
        } else {
            printf("Gradient check passed at index %d:\n", i);
            printf("  Analytical Gradient: %f\n", analytical_gradients[i]);
            printf("  Numerical Gradient: %f\n", numerical_gradient);
        }
    }
}

void print_tensor(const char* name, float* tensor, int size) {
    printf("%s: [", name);
    for (int i = 0; i < size; i++) {
        printf("%.4f%s", tensor[i], (i < size - 1) ? ", " : "");
    }
    printf("]\n");
}


void check_gradients_2d(
    float *plus_perturbed,       // Function values with positive perturbation (1D array, 2D representation)
    float *minus_perturbed,      // Function values with negative perturbation (1D array, 2D representation)
    float *analytical_gradients, // Gradients from backpropagation (1D array, 2D representation)
    int h,                       // Height of the 2D data
    int w,                       // Width of the 2D data
    float epsilon,               // Perturbation value
    float tolerance              // Acceptable tolerance for error
) {
    int size = h * w; // Total number of elements in the 2D representation
    for (int i = 0; i < size; i++) {
        // Compute numerical gradient using finite differences
        float numerical_gradient = (plus_perturbed[i] - minus_perturbed[i]) / (2.0f * epsilon);

        // Compare numerical gradient with analytical gradient
        float difference = fabs(numerical_gradient - analytical_gradients[i]);
        if (difference > tolerance) {
            int row = i / w; // Convert linear index to 2D row
            int col = i % w; // Convert linear index to 2D column
            printf("Gradient check failed at (%d, %d):\n", row, col);
            printf("  Analytical Gradient: %f\n", analytical_gradients[i]);
            printf("  Numerical Gradient: %f\n", numerical_gradient);
            printf("  Difference: %f (tolerance: %f)\n", difference, tolerance);
        } else {
            int row = i / w; // Convert linear index to 2D row
            int col = i % w; // Convert linear index to 2D column
            printf("Gradient check passed at (%d, %d):\n", row, col);
            printf("  Analytical Gradient: %f\n", analytical_gradients[i]);
            printf("  Numerical Gradient: %f\n", numerical_gradient);
        }
    }
}



void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void encoder_enzyme_backward(float *out, int *inp, float *wte, float *dwte, float *wpe, float *dwpe, int B, int T, int C){
    float* d_out = (float*)calloc(B * T * C, sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < C; h++) {
                 for (int i = 0; i < B * T * C; i++) {
                    d_out[i] = 0.0f;
                }
                d_out[b * T * C  + t * C + h] = 1.0f; 
                __enzyme_autodiff(
                    (void*)encoder_forward,
                    enzyme_dup, out, d_out, // Output and its seed gradient
                    enzyme_const, inp,
                    enzyme_dup, wte, dwte,
                    enzyme_dup, wpe, dwpe,
                    enzyme_const, B,
                    enzyme_const, T, enzyme_const, C);
            }
        }
    }
    free(d_out);
}

void test_encoder_backward(){
    int B = 1; // Batch size
    int T = 2; // Sequence length
    int C = 3; // Feature size

    // Allocate memory for input and weights
    float *out = (float*)calloc(B * T * C, sizeof(float));
    int *inp = (int*)malloc(B * T * sizeof(int));
    float *wte = (float*)malloc(10 * C * sizeof(float)); // Assume vocabulary size = 10
    float *dwte = (float*)calloc(10 * C, sizeof(float));
    float *wpe = (float*)malloc(T * C * sizeof(float));
    float *dwpe = (float*)calloc(T * C, sizeof(float));

    // Initialize inputs and weights
    inp[0] = 1;
    inp[1] = 3; // Example token IDs
    for (int i = 0; i < 10 * C; i++) {
        wte[i] = (float)(i + 1) / 10.0f;
    }
    for (int i = 0; i < T * C; i++) {
        wpe[i] = (float)(i + 1) / 5.0f;
    }

    // Call encoder_enzyme_backward
    encoder_enzyme_backward(out, inp, wte, dwte, wpe, dwpe, B, T, C);

    // Print resulting gradients for weights
    printf("dwte:\n");
    for (int i = 0; i < 10 * C; i++) {
        if (i % C == 0 && i != 0) printf("\n");
        printf("%0.4f ", dwte[i]);
    }
    printf("\n\ndwpe:\n");
    for (int i = 0; i < T * C; i++) {
        if (i % C == 0 && i != 0) printf("\n");
        printf("%0.4f ", dwpe[i]);
    }
    printf("\n");

    // Free allocated memory
    free(out);
    free(inp);
    free(wte);
    free(dwte);
    free(wpe);
    free(dwpe);
}

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

void layernorm_forward(float*__restrict out, float*__restrict mean, float*__restrict rstd,
                       float*__restrict inp, float*__restrict weight, float*__restrict bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    // printf("%d %d %d\n", B, T, C);
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float*__restrict x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float*__restrict out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}



void layernorm_backward_enzyme(float* out, float* mean, float* rstd,
                       float* inp, float* dinp, float* weight, float *dweight, float* bias, float *dbias,
                       int B, int T, int C) {

    float* d_out = (float*)calloc(B * T * C, sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < C; h++) {
                for (int i = 0; i < B * T * C; i++) {
                    d_out[i] = 0.0f;
                }
                d_out[b * T * C  + t * C + h] = 1.0f; 

                __enzyme_autodiff(
                    (void*)layernorm_forward,
                    enzyme_dup, out, d_out, // Output and its seed gradient
                    enzyme_const, mean, enzyme_const, rstd,
                    enzyme_dup, inp, dinp,
                    enzyme_dup, weight, dweight,
                    enzyme_dup, bias, dbias,
                    enzyme_const, B,
                    enzyme_const, T, enzyme_const, C);
            }
        }
    }
    free(d_out);
}


void layernorm_backward_enzyme_no_loops(float* out, float* mean, float* rstd,
                       float* inp, float* dinp, float* weight, float *dweight, float* bias, float *dbias,
                       int B, int T, int C) {

    float* d_out = (float*)calloc(B * T * C, sizeof(float));
    // for (int b = 0; b < B; b++) {
    //     for (int t = 0; t < T; t++) {
    //         for (int h = 0; h < C; h++) {
    //             for (int i = 0; i < B * T * C; i++) {
    //                 d_out[i] = 0.0f;
    //             }
    d_out[0] = 1.0f; 

    __enzyme_autodiff(
        (void*)layernorm_forward,
        enzyme_dup, out, d_out, // Output and its seed gradient
        enzyme_const, mean, enzyme_const, rstd,
        enzyme_dup, inp, dinp,
        enzyme_dup, weight, dweight,
        enzyme_dup, bias, dbias,
        enzyme_const, B,
        enzyme_const, T, enzyme_const, C);
    //         }
    //     }
    // }
    free(d_out);
}

// void test_layernorm_backward_enzyme() {
//      const int B = 1, T = 4, C = 3; // No batching, sequence length 4, embedding size 3

//       const float inp[B * T * C] = {1.0f, 2.0f, 3.0f, 
//                                   4.0f, 5.0f, 6.0f, 
//                                   7.0f, 8.0f, 9.0f, 
//                                   10.0f, 11.0f, 12.0f};

//     const float weight[C] = {1.0f, 1.0f, 1.0f};
//     const float bias[C] = {0.0f, 0.0f, 0.0f};

//     // Outputs from forward pass
//     float out[B * T * C];
//     float mean[B * T];
//     float rstd[B * T];

//     // Outputs from backward pass
//     float dinp[B * T * C] = {0}; // Gradient w.r.t. input
//     float dweight[C] = {0};
//     float dbias[C] = {0};

//     // Run forward pass
//     layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);

//     // Print results of forward pass
//     print_tensor("Input", inp, B * T * C);
//     print_tensor("Output", out, B * T * C);
//     print_tensor("Mean", mean, B * T);
//     print_tensor("RSTD", rstd, B * T);

//     // Run backward pass
//     layernorm_backward_enzyme(out, mean, rstd, inp, dinp, weight, dweight, bias, dbias, B, T, C);

//     // Print gradients
//     print_tensor("dInput", dinp, B * T * C);
//     print_tensor("dWeight", dweight, C);
//     print_tensor("dBias", dbias, C);
// }


void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void append_arrays(float *a, float *b, int size_a, int size_b) {
    for (int i = 0; i < size_b; i++) {
        a[size_a + i] = b[i];
    }
}

void matmul_forward_naive(float *__restrict out,
                         const float *__restrict inp, const float *__restrict weight, const float *__restrict bias,
                         int B, int T, int C, int OC) { 
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
   
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}







void matmul_forward(float *__restrict out,
                    const float *__restrict inp, const float *__restrict weight, const float *__restrict bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
   
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

void matmul_backward_enzyme_no_loops(float* out,
                    const float* inp, const float* weight, float *dweight, const float* bias, float *dbias,
                    int B, int T, int C, int OC) {
    float* d_out = (float*)calloc(B * T * OC, sizeof(float));
    d_out[0] = 1.0f; 
    __enzyme_autodiff(
    (void*)matmul_forward,
    enzyme_dup, out, d_out, // Output and its seed gradient
    enzyme_const, inp,  // Input and temporary gradient storage
    enzyme_dup, weight, dweight,
    enzyme_dup, bias, dbias,
    enzyme_const, B,
    enzyme_const, T, enzyme_const, C, enzyme_const, OC);
}


void matmul_backward_naive_enzyme_no_loops(float* out,
                    const float* inp, const float* weight, float *dweight, const float* bias, float *dbias,
                    int B, int T, int C, int OC) {
    float* d_out = (float*)calloc(B * T * OC, sizeof(float));
    d_out[0] = 1.0f; 

    __enzyme_autodiff(
    (void*)matmul_forward_naive,
    enzyme_dup, out, d_out, // Output and its seed gradient
    enzyme_const, inp,  // Input and temporary gradient storage
    enzyme_dup, weight, dweight,
    enzyme_dup, bias, dbias,
    enzyme_const, B,
    enzyme_const, T, enzyme_const, C, enzyme_const, OC);
            
}



void matmul_backward_enzyme(float* out,
                    const float* inp, const float* weight, float *dweight, const float* bias, float *dbias,
                    int B, int T, int C, int OC) {

    float* d_out = (float*)calloc(B * T * OC, sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {    
            for (int o = 0; o < OC; o++) {
                for (int i = 0; i < B * T * OC; i++) {
                    d_out[i] = 0.0f;
                }
                d_out[b * T * OC  + t * OC + o] = 1.0f; 

                 __enzyme_autodiff(
                    (void*)matmul_forward,
                    enzyme_dup, out, d_out, // Output and its seed gradient
                    enzyme_const, inp,  // Input and temporary gradient storage
                    enzyme_dup, weight, dweight,
                    enzyme_dup, bias, dbias,
                    enzyme_const, B,
                    enzyme_const, T, enzyme_const, C, enzyme_const, OC);
            }
        }

    }      
}

void matmul_backward_naive_enzyme(float* out,
                    const float* inp, const float* weight, float *dweight, const float* bias, float *dbias,
                    int B, int T, int C, int OC) {

    float* d_out = (float*)calloc(B * T * OC, sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {    
            for (int o = 0; o < OC; o++) {
                for (int i = 0; i < B * T * OC; i++) {
                    d_out[i] = 0.0f;
                }
                d_out[b * T * OC  + t * OC + o] = 1.0f; 

                 __enzyme_autodiff(
                    (void*)matmul_forward_naive,
                    enzyme_dup, out, d_out, // Output and its seed gradient
                    enzyme_const, inp,  // Input and temporary gradient storage
                    enzyme_dup, weight, dweight,
                    enzyme_dup, bias, dbias,
                    enzyme_const, B,
                    enzyme_const, T, enzyme_const, C, enzyme_const, OC);
            }
        }

    }      
}


void matmul_backward_enzyme_test() {
    int B = 2, T = 3, C = 4, OC = 2;
    // Initialize inputs
    float inp[B * T * C];
    float weight[C * OC];
    float bias[OC];
    float dout[B * T * OC];
    float dweight[C * OC];
    float dbias[OC];

    // Fill inputs with example values
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i + 1);
    for (int i = 0; i < C * OC; i++) weight[i] = 0.01f * (i + 1);
    for (int i = 0; i < OC; i++) bias[i] = 0.0f * (i + 1);
    for (int i = 0; i < B * T * OC; i++) dout[i] = 0.2f * (i + 1);

    // Initialize gradients
    for (int i = 0; i < C * OC; i++) dweight[i] = 0.0f;
    for (int i = 0; i < OC; i++) dbias[i] = 0.0f;

    // Perform backward pass
    matmul_backward_enzyme(dout, inp, weight, dweight, bias, dbias, B, T, C, OC);

    // Print results
    printf("dweight:\n");
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < OC; j++) {
            printf("%f ", dweight[i * OC + j]);
        }
        printf("\n");
    }

    printf("dbias:\n");
    for (int i = 0; i < OC; i++) {
        printf("%f ", dbias[i]);
    }
    printf("\n");
}



void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
   
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                const float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
   
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                const float* dout_bt = dout + b * T * OC + t * OC;
                const float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

void attention_forward(float*__restrict out, float*__restrict preatt, float*__restrict att,
                       float*__restrict inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    // printf("%d %d %d %d\n", B, T, C, NH);
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float*__restrict query_t = inp + b * T * C3 + t * C3 + h * hs;
                float*__restrict preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float*__restrict att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float*__restrict key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float*__restrict out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float*__restrict value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_backward_enzyme(float* out, float* preatt, float* dpreatt, float* att, float *datt, float* inp, float *dinp, int B, int T, int C, int NH) {
    float* d_out = (float*)calloc(B * T * C, sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < C; h++) {
                for (int i = 0; i < B * T * C; i++) {
                    d_out[i] = 0.0f;
                }
                d_out[b * T * C  + t * C + h] = 1.0f; 

                __enzyme_autodiff(
                    (void*)attention_forward,
                    enzyme_dup, out, d_out, // Output and its seed gradient
                    enzyme_dup, preatt, dpreatt,  // Input and temporary gradient storage
                    enzyme_dup, att, datt,
                    enzyme_dup, inp, dinp,
                    enzyme_const, B,
                    enzyme_const, T, enzyme_const, C, enzyme_const, NH);
            }
        }
    }
    free(d_out);
}

void attention_backward_enzyme_no_loops(float* out, float* preatt, float* dpreatt, float* att, float *datt, float* inp, float *dinp, int B, int T, int C, int NH) {
    float* d_out = (float*)calloc(B * T * C, sizeof(float));
    // for (int b = 0; b < B; b++) {
    //     for (int t = 0; t < T; t++) {
    //         for (int h = 0; h < C; h++) {
    //             for (int i = 0; i < B * T * C; i++) {
    //                 d_out[i] = 0.0f;
    //             }
    d_out[0] = 1.0f; 

    __enzyme_autodiff(
        (void*)attention_forward,
        enzyme_dup, out, d_out, // Output and its seed gradient
        enzyme_dup, preatt, dpreatt,  // Input and temporary gradient storage
        enzyme_dup, att, datt,
        enzyme_dup, inp, dinp,
        enzyme_const, B,
        enzyme_const, T, enzyme_const, C, enzyme_const, NH);
    //         }
    //     }
    // }
    free(d_out);
}



void test_attention_backward_enzyme() {
    int T = 4;   // Sequence length
    int C = 3;   // Embedding dimension

    // Input and gradients sizes
    int out_size = T * C;
    int preatt_size = T * T;
    int att_size = preatt_size;
    int inp_size = T * 3 * C;  // 3C for Q, K, V

    // Allocate memory for inputs and gradients
    float* out = (float*)calloc(out_size, sizeof(float));
    float* preatt = (float*)calloc(preatt_size, sizeof(float));
    float* dpreatt = (float*)calloc(preatt_size, sizeof(float));
    float* att = (float*)calloc(att_size, sizeof(float));
    float* datt = (float*)calloc(att_size, sizeof(float));
    float* inp = (float*)calloc(inp_size, sizeof(float));
    float* dinp = (float*)calloc(inp_size, sizeof(float));

    // Initialize inputs with dummy data
    for (int i = 0; i < inp_size; i++) {
        inp[i] = (float)(i % 5) * 0.1f;  // Dummy data
    }
    for (int i = 0; i < att_size; i++) {
        att[i] = (float)(i % 3) * 0.2f;  // Dummy data
    }

    // Call the backward function
    attention_backward_enzyme(out, preatt, dpreatt, att, datt, inp, dinp, 1, T, C, 1);

    // Print gradients
    print_tensor("Gradients for preatt", dpreatt, preatt_size);
    print_tensor("Gradients for att", datt, att_size);
    print_tensor("Gradients for inp", dinp, inp_size);

    // Free memory
    free(out);
    free(preatt);
    free(dpreatt);
    free(att);
    free(datt);
    free(inp);
    free(dinp);
}


void attention_backward(float* dinp, float* dpreatt, float* datt,
                        float* dout, float* inp, float* att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.f / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

void gelu_enzyme_backward(float *out, float *inp, float *dinp, int N) {
    float* d_out = (float*)calloc(N, sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            d_out[j] = 0.0f;
        }
        d_out[i] = 1.0f;
        __enzyme_autodiff(
                    (void*)gelu_forward,
                    enzyme_dup, out, d_out, // Output and its seed gradient
                    enzyme_dup, inp, dinp,
                    enzyme_const, N);
    }
    free(d_out);
}

void test_gelu_enzyme_backward(){
     int N = 5; // Example size

    // Allocate memory
    float *inp = (float*)malloc(N * sizeof(float));
    float *out = (float*)malloc(N * sizeof(float));
    float *dinp = (float*)calloc(N, sizeof(float)); // Gradients of input

    // Initialize inputs
    for (int i = 0; i < N; i++) {
        inp[i] = (float)(i + 1); // Example values: [1.0, 2.0, 3.0, 4.0, 5.0]
    }

    // Call gelu_enzyme_backward
    gelu_enzyme_backward(out, inp, dinp, N);

    // Print results
    printf("dinp (gradients of input):\n");
    for (int i = 0; i < N; i++) {
        printf("dinp[%d]: %0.4f\n", i, dinp[i]);
    }

    // Free memory
    free(inp);
    free(out);
    free(dinp);
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}


void risidual_enzyme_backward(float *out, float *inp1, float *dinp1, float *inp2, float *dinp2, int N) {
    float* d_out = (float*)calloc(N, sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            d_out[j] = 0.0f;
        }
        d_out[i] = 1.0f;
        __enzyme_autodiff(
                    (void*)residual_forward,
                    enzyme_dup, out, d_out, // Output and its seed gradient
                    enzyme_dup, inp1, dinp1,
                    enzyme_dup, inp2, dinp2,
                    enzyme_const, N);
    }
    free(d_out);
}

void test_risidual_enzyme_backward() {
    int N = 5; // Example size

    // Allocate memory
    float *inp1 = (float*)malloc(N * sizeof(float));
    float *inp2 = (float*)malloc(N * sizeof(float));
    float *out = (float*)malloc(N * sizeof(float));
    float *dinp1 = (float*)calloc(N, sizeof(float)); // Gradients of inp1
    float *dinp2 = (float*)calloc(N, sizeof(float)); // Gradients of inp2

    // Initialize inputs
    for (int i = 0; i < N; i++) {
        inp1[i] = (float)(i + 1);       // Example values: [1.0, 2.0, 3.0, 4.0, 5.0]
        inp2[i] = (float)(5 - i);       // Example values: [5.0, 4.0, 3.0, 2.0, 1.0]
    }

    // Call residual_enzyme_backward
    risidual_enzyme_backward(out, inp1, dinp1, inp2, dinp2, N);

    // Print forward output
    printf("Residual output (forward pass):\n");
    for (int i = 0; i < N; i++) {
        printf("out[%d]: %0.4f\n", i, out[i]);
    }

    // Print gradients
    printf("\nGradients of inp1 (dinp1):\n");
    for (int i = 0; i < N; i++) {
        printf("dinp1[%d]: %0.4f\n", i, dinp1[i]);
    }

    printf("\nGradients of inp2 (dinp2):\n");
    for (int i = 0; i < N; i++) {
        printf("dinp2[%d]: %0.4f\n", i, dinp2[i]);
    }

    // Free memory
    free(inp1);
    free(inp2);
    free(out);
    free(dinp1);
    free(dinp2);
}

int get_4d_index(int x, int y, int z, int w, int dim1, int dim2, int dim3, int dim4) {
    return x * dim2 * dim3 * dim4 + y * dim3 * dim4 + z * dim4 + w;
}

int get_2d_index(int x, int y, int dim1, int dim2) {
    return x * dim2 + y;
}


void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
   
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // Compute the base index for this (b, t) pair
            int base_index = b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits[base_index + i] > maxval) {
                    maxval = logits[base_index + i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs[base_index + i] = expf(logits[base_index + i] - maxval);
                sum += probs[base_index + i];
            }
            // Normalize the probabilities
            for (int i = 0; i < V; i++) {
                probs[base_index + i] /= sum;
            }
            // Zero out the padded dimensions
            for (int i = V; i < Vp; i++) {
                probs[base_index + i] = 0.0f;
            }
        }
    }
}

void softmax_backward_enzyme(float* probs, float* logits, float* d_logits, int B, int T, int V, int Vp) {
    // Zero-initialize gradients for logits
    // returns jacobian marix of partial softmax / partial logits (d_logtis, size Vp * Vp)

    // Allocate temporary storage for d_logits
    float* d_probs = (float*)calloc(B * T * Vp, sizeof(float));
    float* temp_d_logits = (float*)calloc(B * T * Vp, sizeof(float)); // Temporary storage for each gradient computation

    // Loop over each output element (losses[b][t])
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int v = 0; v < Vp; v++) {
                // Reset d_probs for this specific loss
                for (int i = 0; i < B * T * Vp; i++) {
                    d_probs[i] = 0.0f;
                }
                d_probs[b * T * Vp + t * Vp + v] = 1.0f; // Seed for this specific loss
                // Reset temp_d_logits for this computation
                for (int i = 0; i < B * T * Vp; i++) {
                    temp_d_logits[i] = 0.0f;
                }

                // Call Enzyme to compute gradient for this output
                __enzyme_autodiff(
                    (void*)softmax_forward,
                    enzyme_dup, probs, d_probs, // Output and its seed gradient
                    enzyme_dup, logits, temp_d_logits,  // Input and temporary gradient storage
                    enzyme_const, B,
                    enzyme_const, T, enzyme_const, V, enzyme_const, Vp);

                // copy logits over
                for (int i = 0; i < Vp; i++){
                    d_logits[get_4d_index(b, t, v, i, B, T, Vp, Vp)] = temp_d_logits[i];
                }
            }
        }
    }

    free(d_probs);
    free(temp_d_logits);
}


void test_softmax_enzyme() {
    const int B = 1, T = 1, Vp = 4, V = 4;
    float logits[B * T * Vp] = {
        10.0f, 4.0f, 0.0f, 5.0f
    };

    float d_logits[B * T * Vp * Vp] = {0.0};
    float probs[B * T * Vp] = {0.0};
    float analytical_gradient[Vp * Vp] = {0.0}; // for storing the analytical dimension for 1 example

    // Compute gradients
    softmax_backward_enzyme(probs, logits, d_logits, B, T, V, Vp);
    // softmax_forward(probs, logits, B, T, V, Vp);

    // Print gradients
    printf("Gradients:\n");
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int v = 0; v < Vp; v++) {
                for (int w = 0; w < Vp; w++) {
                    printf("%f ", d_logits[get_4d_index(b, t, v, w, B, T, Vp, Vp)]);
                    analytical_gradient[get_2d_index(v, w, Vp, Vp)] = d_logits[get_4d_index(b, t, v, w, B, T, Vp, Vp)];
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }

    // do finite distance check
     // Perturb each probability and compute numerical gradients
    float perturbed_plus[Vp * Vp] = {0.0};
    float perturbed_minus[Vp * Vp] = {0.0};
    for (int i = 0; i < Vp; i++) {
        float original_logit = logits[i];

        // Perturb the probability by +EPSILON
        logits[i] = original_logit + EPSILON;
        float perturbed_probs_plus[Vp] = {0.0};
        softmax_forward(perturbed_probs_plus, logits, B, T, V, Vp);

        // Perturb the probability by -EPSILON
        logits[i] = original_logit - EPSILON;
        float perturbed_probs_minus[Vp] = {0.0};
        softmax_forward(perturbed_probs_minus, logits, B, T, V, Vp);
            

        append_arrays(perturbed_plus, perturbed_probs_plus, i * Vp, Vp);
        append_arrays(perturbed_minus, perturbed_probs_minus, i * Vp, Vp);
        // Restore original probability
        logits[i] = original_logit;
    }


    check_gradients_2d(perturbed_plus, perturbed_minus, analytical_gradient, Vp, Vp, EPSILON, TOLERANCE);

    // Check gradients
    // printf("Checking gradients:\n");
    // check_gradients(numerical_gradients, d_probs, B * T * Vp, EPSILON, TOLERANCE);
    printf("\n");
}





void crossentropy_forward(float*__restrict losses,
                          float*__restrict probs, int*__restrict targets,
                          int B, int T, int Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float*__restrict probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}



void crossentropy_backward_enzyme(float* probs, float* d_probs,
                                    float* losses, int* targets,
                                    int B, int T, int Vp) {
    // Zero-initialize gradients for probs
    for (int i = 0; i < B * T * Vp; i++) {
        d_probs[i] = 0.0f;
    }

    // Allocate temporary storage for d_losses
   float* d_losses = (float*)calloc(B * T, sizeof(float));
    for (int i = 0; i < B * T; i++) {
        d_losses[i] = 0.0f; // Initialize to zero
    }

    // Loop over each output element (losses[b][t])
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // Reset d_losses
            for (int i = 0; i < B * T; i++) {
                d_losses[i] = 0.0f;
            }
            d_losses[b * T + t] = 1.0f; // Seed for this specific loss

            // Call Enzyme to compute gradient for this output
            __enzyme_autodiff(
                (void*)crossentropy_forward,
                enzyme_dup, losses, d_losses, // Output and its seed gradient
                enzyme_dup, probs, d_probs,  // Input and gradient
                enzyme_const, targets,
                enzyme_const, B, enzyme_const, T, enzyme_const, Vp);
        }
    }

    free(d_losses);
}

void crossentropy_backward_enzyme_no_loops(float* probs, float* d_probs,
                                    float* losses, int* targets,
                                    int B, int T, int Vp) {
    // Zero-initialize gradients for probs
    // for (int i = 0; i < B * T * Vp; i++) {
    //     d_probs[i] = 0.0f;
    // }

    // Allocate temporary storage for d_losses
    float* d_losses = (float*)calloc(B * T, sizeof(float));
    // for (int i = 0; i < B * T; i++) {
    //     d_losses[i] = 0.0f; // Initialize to zero
    // }

    // Loop over each output element (losses[b][t])
    // for (int b = 0; b < B; b++) {
    //     for (int t = 0; t < T; t++) {
    //         // Reset d_losses
    //         for (int i = 0; i < B * T; i++) {
    //             d_losses[i] = 0.0f;
    //         }
    d_losses[0] = 1.0f; // Seed for this specific loss

    // Call Enzyme to compute gradient for this output
    __enzyme_autodiff(
        (void*)crossentropy_forward,
        enzyme_dup, losses, d_losses, // Output and its seed gradient
        enzyme_dup, probs, d_probs,  // Input and gradient
        enzyme_const, targets,
        enzyme_const, B, enzyme_const, T, enzyme_const, Vp);
    //     }
    // }

    free(d_losses);
}



// void test_crossentropy_enzyme() {
//     const int B = 1, T = 1, Vp = 4;
//     float probs[B * T * Vp] = {
//         0.1, 0.2, 0.3, 0.4,  // Batch 0, Time 0
//     };

//     int targets[B * T] = {
//         1, 0, 0,
//     };
//     float losses[B * T] = {0.0};
//     float d_probs[B * T * Vp] = {0.0};

//     // Compute gradients
//     crossentropy_backward_enzyme(probs, d_probs, losses, targets, B, T, Vp);

//     // Print gradients
//     printf("Gradients:\n");
//     for (int b = 0; b < B; b++) {
//         for (int t = 0; t < T; t++) {
//             for (int v = 0; v < Vp; v++) {
//                 printf("%f ",
//                        d_probs[b * T * Vp + t * Vp + v]);
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }

//     // do finite distance check
//      // Perturb each probability and compute numerical gradients
//     float numerical_gradients[B * T * Vp] = {0.0};
//     for (int i = 0; i < B * T * Vp; i++) {
//         float original_prob = probs[i];

//         // Perturb the probability by +EPSILON
//         probs[i] = original_prob + EPSILON;
//         float perturbed_losses_plus[B * T] = {0.0};
//         crossentropy_forward(perturbed_losses_plus, probs, targets, B, T, Vp);

//         // Perturb the probability by -EPSILON
//         probs[i] = original_prob - EPSILON;
//         float perturbed_losses_minus[B * T] = {0.0};
//         crossentropy_forward(perturbed_losses_minus, probs, targets, B, T, Vp);

//         // Restore original probability
//         probs[i] = original_prob;

//         // Compute numerical gradient as (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
//         numerical_gradients[i] = (perturbed_losses_plus[0] - perturbed_losses_minus[0]) / (2 * EPSILON);
//     }

//     // Check gradients
//     printf("Checking gradients:\n");
//     check_gradients(numerical_gradients, d_probs, B * T * Vp, EPSILON, TOLERANCE);
//     printf("\n");
// }
void crossentropy_softmax_backward_enzyme(float* dlogits,
                                          float* dlosses,
                                          float* probs,
                                          int* targets,
                                          int B, int T, int V, int Vp) {
    // Allocate memory for intermediate gradients
    float* d_probs = (float*)calloc(B * T * Vp, sizeof(float));

    // Backprop through cross-entropy to compute d_probs
    crossentropy_backward_enzyme(probs, d_probs, dlosses, targets, B, T, Vp);

    // Backprop through softmax to compute dlogits
    softmax_backward_enzyme(probs, probs, dlogits, B, T, V, Vp);

    // Accumulate gradients from d_probs into dlogits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int v = 0; v < Vp; v++) {
                dlogits[b * T * Vp + t * Vp + v] += d_probs[b * T * Vp + t * Vp + v];
            }
        }
    }

    // Free allocated memory
    free(d_probs);
}



void test_crossentropy_softmax_backward_enzyme() {
    const int B = 1, T = 1, V = 3, Vp = 3; // Batch size, time steps, vocab size
    float logits[Vp] = {2.0f, 1.0f, 0.1f}; // Example logits
    float probs[V] = {0.0};
    softmax_forward(probs, logits, B, T, V, Vp);

    int targets[1] = {1}; // Target class index
    float dlosses[1] = {1.0f}; // Gradient of loss (dL/dloss)

    float dlogits[Vp] = {0.0}; // Output storage
    for (int i = 0; i < Vp; i++) dlogits[i] = 0.0f;

    // Run backward pass
    crossentropy_softmax_backward_enzyme(dlogits, dlosses, probs, targets, B, T, V, Vp);

    // Print results
    printf("Softmax probabilities:\n");
    for (int i = 0; i < V; i++) {
        printf("probs[%d] = %.6f\n", i, probs[i]);
    }

    printf("\nGradients of logits (dlogits):\n");
    for (int i = 0; i < Vp; i++) {
        printf("dlogits[%d] = %.6f\n", i, dlogits[i]);
    }

}

void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}




void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        model->targets = (int*)mallocCheck(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }
    
    crossentropy_backward_enzyme(acts.probs, grads_acts.logits, grads_acts.losses, model->targets, B, T, Vp);
    // crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}



void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}


// BENCHMARKING

// MATMUL NAIVE
static void BM_matmul_backward_scaled(benchmark::State& state) {
    int B = state.range(0); // Batch size
    int T = state.range(1); // Time steps
    int C = state.range(2); // Input channels
    int OC = state.range(3); // Output channels

    // Allocate data
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));
    float *__restrict weight = (float*)malloc(C * OC * sizeof(float));
    float *__restrict bias = (float*)malloc(OC * sizeof(float));
    float *__restrict dout = (float*)malloc(B * T * OC * sizeof(float));

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C * OC; i++) weight[i] = 0.01f * (i % 50 + 1);
    for (int i = 0; i < OC; i++) bias[i] = 0.0f;
    for (int i = 0; i < B * T * OC; i++) dout[i] = 0.2f * (i % 200 + 1);

    // Allocate outputs and gradients
    float *__restrict dinp = (float*)calloc(B * T * C, sizeof(float));
    float *__restrict dweight = (float*)calloc(C * OC, sizeof(float));
    float *__restrict dbias = (float*)calloc(OC, sizeof(float));

    // Benchmark the matmul_backward function
    for (auto _ : state) {
        matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
        benchmark::DoNotOptimize(dinp);
        benchmark::DoNotOptimize(dweight);
        benchmark::DoNotOptimize(dbias);
    }

    // Free memory
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
}


static void BM_matmul_forward_scaled(benchmark::State& state) {
    int B = state.range(0); // Batch size
    int T = state.range(1); // Time steps
    int C = state.range(2); // Input channels
    int OC = state.range(3); // Output channels

    // Allocate data
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));
    float *__restrict weight = (float*)malloc(C * OC * sizeof(float));
    float *__restrict bias = (float*)malloc(OC * sizeof(float));
    float *__restrict out = (float*)malloc(B * T * OC * sizeof(float));

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C * OC; i++) weight[i] = 0.01f * (i % 50 + 1);
    for (int i = 0; i < OC; i++) bias[i] = 0.0f;

    // Benchmark the matmul_forward function
    for (auto _ : state) {
        matmul_forward(out, inp, weight, bias, B, T, C, OC);
        benchmark::DoNotOptimize(out);
    }

    // Free memory
    free(inp);
    free(weight);
    free(bias);
    free(out);
}

static void BM_matmul_forward_naive_scaled(benchmark::State& state) {
    int B = state.range(0); // Batch size
    int T = state.range(1); // Time steps
    int C = state.range(2); // Input channels
    int OC = state.range(3); // Output channels

    // Allocate data
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));
    float *__restrict weight = (float*)malloc(C * OC * sizeof(float));
    float *__restrict bias = (float*)malloc(OC * sizeof(float));
    float *__restrict out = (float*)malloc(B * T * OC * sizeof(float));

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C * OC; i++) weight[i] = 0.01f * (i % 50 + 1);
    for (int i = 0; i < OC; i++) bias[i] = 0.0f;

    // Benchmark the matmul_forward function
    for (auto _ : state) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        benchmark::DoNotOptimize(out);
    }

    // Free memory
    free(inp);
    free(weight);
    free(bias);
    free(out);
}



static void BM_matmul_naive_backward_enzyme_scaled(benchmark::State& state) {
    int B = state.range(0); // Batch size
    int T = state.range(1); // Time steps
    int C = state.range(2); // Input channels
    int OC = state.range(3); // Output channels

    // Allocate data
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));
    float *__restrict weight = (float*)malloc(C * OC * sizeof(float));
    float *__restrict bias = (float*)malloc(OC * sizeof(float));
    float* dout = (float*)malloc(B * T * OC * sizeof(float));

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C * OC; i++) weight[i] = 0.01f * (i % 50 + 1);
    for (int i = 0; i < OC; i++) bias[i] = 0.0f;
    for (int i = 0; i < B * T * OC; i++) dout[i] = 0.2f * (i % 200 + 1);

    // Allocate outputs and gradients
    float *__restrict enzyme_out = (float*)calloc(B * T * OC, sizeof(float));
    float *__restrict dweight = (float*)calloc(C * OC, sizeof(float));
    float *__restrict dbias = (float*)calloc(OC, sizeof(float));

    // Benchmark the enzyme implementation
    for (auto _ : state) {
        matmul_backward_naive_enzyme_no_loops(enzyme_out, inp, weight, dweight, bias, dbias, B, T, C, OC);
        benchmark::DoNotOptimize(enzyme_out);
        benchmark::DoNotOptimize(dweight);
        benchmark::DoNotOptimize(dbias);
    }

    // Free memory
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(enzyme_out);
    free(dweight);
    free(dbias);
}

static void BM_matmul_backward_enzyme_scaled(benchmark::State& state) {
    int B = state.range(0); // Batch size
    int T = state.range(1); // Time steps
    int C = state.range(2); // Input channels
    int OC = state.range(3); // Output channels

    // Allocate data
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));
    float *__restrict weight = (float*)malloc(C * OC * sizeof(float));
    float *__restrict bias = (float*)malloc(OC * sizeof(float));
    float *__restrict dout = (float*)malloc(B * T * OC * sizeof(float));

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C * OC; i++) weight[i] = 0.01f * (i % 50 + 1);
    for (int i = 0; i < OC; i++) bias[i] = 0.0f;
    for (int i = 0; i < B * T * OC; i++) dout[i] = 0.2f * (i % 200 + 1);

    // Allocate outputs and gradients
    float *__restrict enzyme_out = (float*)calloc(B * T * OC, sizeof(float));
    float *__restrict dweight = (float*)calloc(C * OC, sizeof(float));
    float *__restrict dbias = (float*)calloc(OC, sizeof(float));

    // Benchmark the enzyme implementation
    for (auto _ : state) {
        matmul_backward_enzyme_no_loops(enzyme_out, inp, weight, dweight, bias, dbias, B, T, C, OC);
        benchmark::DoNotOptimize(enzyme_out);
        benchmark::DoNotOptimize(dweight);
        benchmark::DoNotOptimize(dbias);
    }

    // Free memory
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(enzyme_out);
    free(dweight);
    free(dbias);
}


static void BM_matmul_backward_naive_enzyme_scaled(benchmark::State& state) {
    int B = state.range(0); // Batch size
    int T = state.range(1); // Time steps
    int C = state.range(2); // Input channels
    int OC = state.range(3); // Output channels

    // Allocate data
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));
    float *__restrict weight = (float*)malloc(C * OC * sizeof(float));
    float *__restrict bias = (float*)malloc(OC * sizeof(float));
    float *__restrict dout = (float*)malloc(B * T * OC * sizeof(float));

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C * OC; i++) weight[i] = 0.01f * (i % 50 + 1);
    for (int i = 0; i < OC; i++) bias[i] = 0.0f;
    for (int i = 0; i < B * T * OC; i++) dout[i] = 0.2f * (i % 200 + 1);

    // Allocate outputs and gradients
    float *__restrict enzyme_out = (float*)calloc(B * T * OC, sizeof(float));
    float *__restrict dweight = (float*)calloc(C * OC, sizeof(float));
    float *__restrict dbias = (float*)calloc(OC, sizeof(float));

    // Benchmark the enzyme implementation
    for (auto _ : state) {
        matmul_backward_naive_enzyme_no_loops(enzyme_out, inp, weight, dweight, bias, dbias, B, T, C, OC);
        benchmark::DoNotOptimize(enzyme_out);
        benchmark::DoNotOptimize(dweight);
        benchmark::DoNotOptimize(dbias);
    }

    // Free memory
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(enzyme_out);
    free(dweight);
    free(dbias);
}


// ATTENTION
// Benchmark for attention_backward_enzyme
static void BM_attention_backward_enzyme_scaled(benchmark::State& state) {
    int B = state.range(0); // Batch size
    int T = state.range(1); // Time steps
    int C = state.range(2); // Input channels
    int NH = state.range(3); // Number of heads

    // Allocate memory
    float *__restrict out = (float*)malloc(B * T * C * sizeof(float));            // Output (B, T, C)
    float *__restrict preatt = (float*)malloc(B * NH * T * T * sizeof(float));   // Pre-attention (B, NH, T, T)
    float *__restrict dpreatt = (float*)malloc(B * NH * T * T * sizeof(float));  // Gradient of pre-attention (B, NH, T, T)
    float *__restrict att = (float*)malloc(B * NH * T * T * sizeof(float));      // Attention (B, NH, T, T)
    float *__restrict datt = (float*)malloc(B * NH * T * T * sizeof(float));     // Gradient of attention (B, NH, T, T)
    float *__restrict inp = (float*)malloc(B * T * 3 * C * sizeof(float));       // Input (B, T, 3C) for Q, K, V
    float *__restrict dinp = (float*)calloc(B * T * 3 * C, sizeof(float));       // Gradient of input (B, T, 3C)


    // Initialize inputs
    for (int i = 0; i < B * T * NH; i++) {
        preatt[i] = 0.1f * (i % 100 + 1);
        dpreatt[i] = 0.1f * (i % 100 + 1);
        att[i] = 0.1f * (i % 50 + 1);
        datt[i] = 0.1f * (i % 50 + 1);
    }
    for (int i = 0; i < B * T * C; i++) {
        inp[i] = 0.1f * (i % 100 + 1);
    }

    // Benchmark
    for (auto _ : state) {
        attention_backward_enzyme_no_loops(out, preatt, dpreatt, att, datt, inp, dinp, B, T, C, NH);
        benchmark::DoNotOptimize(out);
        benchmark::DoNotOptimize(dinp);
    }

    // Free memory
    free(out);
    free(preatt);
    free(dpreatt);
    free(att);
    free(datt);
    free(inp);
    free(dinp);
}

// Benchmark for attention forward
static void BM_attention_forward_scaled(benchmark::State& state) {
    int B = state.range(0);  // Batch size
    int T = state.range(1);  // Sequence length
    int C = state.range(2);  // Input channels
    int NH = state.range(3); // Number of heads

    // Allocate memory
    float *__restrict inp = (float*)malloc(B * T * 3 * C * sizeof(float)); // 3C for Q, K, V
    float *__restrict preatt = (float*)malloc(B * NH * T * T * sizeof(float)); // NH * T * T
    float *__restrict att = (float*)malloc(B * NH * T * T * sizeof(float));   // NH * T * T
    float *__restrict out = (float*)malloc(B * T * C * sizeof(float));        // Output (B, T, C)

    // Initialize inputs
    for (int i = 0; i < B * T * 3 * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < B * NH * T * T; i++) preatt[i] = 0.2f * (i % 50 + 1);
    for (int i = 0; i < B * NH * T * T; i++) att[i] = 0.3f * (i % 30 + 1);

    // Benchmarking loop
    for (auto _ : state) {
        attention_forward(out, preatt, att, inp, B, T, C, NH);
        benchmark::DoNotOptimize(out);
    }

    // Free memory
    free(inp);
    free(preatt);
    free(att);
    free(out);
}

// LAYER NORM

static void BM_layernorm_forward_scaled(benchmark::State& state) {
    int B = state.range(0);  // Batch size
    int T = state.range(1);  // Sequence length
    int C = state.range(2);  // Input channels (feature dimension)

    // Allocate memory
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));   // Input activations (B, T, C)
    float *__restrict out = (float*)malloc(B * T * C * sizeof(float));  // Output activations (B, T, C)
    float *__restrict mean = (float*)malloc(B * T * sizeof(float));     // Mean (B, T)
    float *__restrict rstd = (float*)malloc(B * T * sizeof(float));     // Reciprocal standard deviation (B, T)
    float *__restrict weight = (float*)malloc(C * sizeof(float));       // Scaling parameters (C)
    float *__restrict bias = (float*)malloc(C * sizeof(float));         // Shifting parameters (C)

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C; i++) weight[i] = 1.0f;  // Scale initialized to 1
    for (int i = 0; i < C; i++) bias[i] = 0.0f;    // Bias initialized to 0

    // Benchmarking loop
    for (auto _ : state) {
        layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);
        benchmark::DoNotOptimize(out);
        benchmark::DoNotOptimize(mean);
        benchmark::DoNotOptimize(rstd);
    }

    // Free memory
    free(inp);
    free(out);
    free(mean);
    free(rstd);
    free(weight);
    free(bias);
}

static void BM_layernorm_backward_enzyme_scaled(benchmark::State& state) {
    int B = state.range(0);  // Batch size
    int T = state.range(1);  // Sequence length
    int C = state.range(2);  // Input channels (feature dimension)

    // Allocate memory
    float *__restrict inp = (float*)malloc(B * T * C * sizeof(float));      // Input activations (B, T, C)
    float *__restrict mean = (float*)malloc(B * T * sizeof(float));         // Mean (B, T)
    float *__restrict rstd = (float*)malloc(B * T * sizeof(float));         // Reciprocal standard deviation (B, T)
    float *__restrict weight = (float*)malloc(C * sizeof(float));           // Scaling parameters (C)
    float *__restrict bias = (float*)malloc(C * sizeof(float));             // Shifting parameters (C)
    float *__restrict out = (float*)malloc(B * T * C * sizeof(float));      // Output activations (B, T, C)
    float *__restrict dinp = (float*)malloc(B * T * C * sizeof(float));     // Gradient wrt input (B, T, C)
    float *__restrict dweight = (float*)malloc(C * sizeof(float));          // Gradient wrt weight (C)
    float *__restrict dbias = (float*)malloc(C * sizeof(float));            // Gradient wrt bias (C)

    // Initialize inputs
    for (int i = 0; i < B * T * C; i++) inp[i] = 0.1f * (i % 100 + 1);
    for (int i = 0; i < C; i++) weight[i] = 1.0f;  // Scale initialized to 1
    for (int i = 0; i < C; i++) bias[i] = 0.0f;    // Bias initialized to 0
    for (int i = 0; i < B * T * C; i++) dinp[i] = 0.0f; // Initialize gradient wrt input to 0
    for (int i = 0; i < C; i++) dweight[i] = 0.0f; // Initialize gradient wrt weight to 0
    for (int i = 0; i < C; i++) dbias[i] = 0.0f;   // Initialize gradient wrt bias to 0

    // Benchmarking loop
    for (auto _ : state) {
        layernorm_backward_enzyme_no_loops(out, mean, rstd, inp, dinp, weight, dweight, bias, dbias, B, T, C);
        benchmark::DoNotOptimize(out);
        benchmark::DoNotOptimize(dinp);
        benchmark::DoNotOptimize(dweight);
        benchmark::DoNotOptimize(dbias);
    }

    // Free memory
    free(inp);
    free(mean);
    free(rstd);
    free(weight);
    free(bias);
    free(out);
    free(dinp);
    free(dweight);
    free(dbias);
}


// CROSSENTROPY
static void BM_crossentropy_forward_scaled(benchmark::State& state) {
    int B = state.range(0);  // Batch size
    int T = state.range(1);  // Sequence length
    int Vp = state.range(2); // Vocabulary size

    // Allocate memory
    float *__restrict probs = (float*)malloc(B * T * Vp * sizeof(float)); // Probabilities (B, T, Vp)
    int *__restrict targets = (int*)malloc(B * T * sizeof(int));          // Target indices (B, T)
    float *__restrict losses = (float*)malloc(B * T * sizeof(float));     // Losses (B, T)

    // Initialize inputs
    for (int i = 0; i < B * T * Vp; i++) probs[i] = 1.0f / Vp; // Uniform probabilities
    for (int i = 0; i < B * T; i++) targets[i] = i % Vp;       // Cyclic target indices
    for (int i = 0; i < B * T; i++) losses[i] = 0.0f;          // Initialize losses to 0

    // Benchmarking loop
    for (auto _ : state) {
        crossentropy_forward(losses, probs, targets, B, T, Vp);
        benchmark::DoNotOptimize(losses);
    }

    // Free memory
    free(probs);
    free(targets);
    free(losses);
}

static void BM_crossentropy_backward_enzyme_scaled(benchmark::State& state) {
    int B = state.range(0);  // Batch size
    int T = state.range(1);  // Sequence length
    int Vp = state.range(2); // Vocabulary size

    // Allocate memory
    float *__restrict probs = (float*)malloc(B * T * Vp * sizeof(float));      // Probabilities (B, T, Vp)
    float *__restrict d_probs = (float*)malloc(B * T * Vp * sizeof(float));   // Gradients w.r.t. probs (B, T, Vp)
    float *__restrict losses = (float*)malloc(B * T * sizeof(float));         // Losses (B, T)
    int *__restrict targets = (int*)malloc(B * T * sizeof(int));              // Target indices (B, T)

    // Initialize inputs
    for (int i = 0; i < B * T * Vp; i++) probs[i] = 1.0f / Vp; // Uniform probabilities
    for (int i = 0; i < B * T; i++) targets[i] = i % Vp;       // Cyclic target indices
    for (int i = 0; i < B * T; i++) losses[i] = 0.0f;          // Initialize losses to 0
    for (int i = 0; i < B * T * Vp; i++) d_probs[i] = 0.0f;    // Initialize gradients to 0

    // Benchmarking loop
    for (auto _ : state) {
        crossentropy_backward_enzyme_no_loops(probs, d_probs, losses, targets, B, T, Vp);
        benchmark::DoNotOptimize(d_probs);
    }

    // Free memory
    free(probs);
    free(d_probs);
    free(losses);
    free(targets);
}





// Register the benchmarks
// BENCHMARK(BM_attention_backward_enzyme_scaled)
//     ->Args({16, 128, 512, 8}) // Example: Batch=16, Time=128, Channels=512, Heads=8
//     ->Args({32, 64, 256, 4}); // Example: Batch=32, Time=64, Channels=256, Heads=4

// BENCHMARK(BM_attention_backward_scaled)
//     ->Args({16, 128, 512, 8})
//     ->Args({32, 64, 256, 4});

// MATMUL



void benchmark_matmul_forward(){
    // Register the scaled benchmark with parameters
    BENCHMARK(BM_matmul_forward_naive_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 768}) // Default
        ->Threads(1);
        // ->Threads(12)              // Single-threaded
        // ->Threads(24);


    // BENCHMARK(BM_matmul_backward_enzyme_scaled)
    //     ->Iterations(500)
    //     ->Args({4, 64, 768, 768})
    //     ->Threads(1);  // Default
    //     // ->Threads(12)              // Single-threaded
    //     // ->Threads(24);


    // // Naive
    BENCHMARK(BM_matmul_forward_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 768}) // Default
        ->Threads(1);
        // ->Threads(12)              // Single-threaded
        // ->Threads(24);

    // Not Naive
    BENCHMARK(BM_matmul_backward_enzyme_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 768}) // Default
        ->Threads(1);

    BENCHMARK(BM_matmul_backward_naive_enzyme_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 768}) // Default
        ->Threads(1);
        // ->Threads(12)              // Single-threaded
        // ->Threads(24);
}

void benchmark_attention_forward(){
    BENCHMARK(BM_attention_forward_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 12}) // Default
        ->Threads(1);
    BENCHMARK(BM_attention_backward_enzyme_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 12}) // Default
        ->Threads(1);
}

void benchmark_attention_forward_omp(){
    BENCHMARK(BM_attention_forward_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 12}) // Default
        ->Threads(24)
        ->Threads(48);
    BENCHMARK(BM_attention_backward_enzyme_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768, 12}) // Default
        ->Threads(24)
        ->Threads(48);
}



void benchmark_layernorm_forward(){
    BENCHMARK(BM_layernorm_forward_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768}) // Default
        ->Threads(1);
    BENCHMARK(BM_layernorm_backward_enzyme_scaled)
        ->Iterations(500)
        ->Args({4, 64, 768}) // Default
        ->Threads(1);
}




void benchmark_crossentropy_forward(){
    BENCHMARK(BM_crossentropy_forward_scaled)
        ->Iterations(500)
        ->Args({4, 64, 50304}) // Default
        ->Threads(1);
    BENCHMARK(BM_crossentropy_backward_enzyme_scaled)
        ->Iterations(500)
        ->Args({4, 64, 50304}) // Default
        ->Threads(1);
}



// void benchmark_attention_forward(){
//     BENCHMARK(BM_attention_forward_scaled)
//         ->Iterations(500)
//         ->Args({4, 64, 768, 12}) // Default
//         ->Threads(1);
//     BENCHMARK(BM_attention_backward_enzyme_scaled)
//         ->Iterations(500)
//         ->Args({4, 64, 768, 12}) // Default
//         ->Threads(1);
// }

int main(int argc, char** argv) {
    benchmark_attention_forward_omp();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
// ----------------------------------------------------------------------------
// main training loop
int main_2() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}
#endif