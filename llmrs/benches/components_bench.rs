#![feature(autodiff)]
use std::autodiff::autodiff;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};


#[autodiff(matmul_naive_backward, Reverse, Duplicated, Const, Duplicated, Duplicated, Const, Const, Const, Const)]
fn matmul_forward_naive(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {
    for bt in 0..(b * t) {
        for o in 0..oc {
            let mut sum = bias[o];

            for i in 0..c {
                sum += inp[bt * c + i] * weight[i + o * c];
            }

            out[bt * oc + o] = sum;
        }
    }
}


#[autodiff(matmul_backward, Reverse, Duplicated, Const, Duplicated, Duplicated, Const, Const, Const, Const)]
fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {
    const LOOP_UNROLL: usize = 8;
    let bt = b * t;

    if bt % LOOP_UNROLL != 0 {
        matmul_forward_naive(out, inp, weight, bias, b, t, c, oc);
        return;
    }

    // Iterate over the collapsed B and T dimensions in strides of LOOP_UNROLL
    for obt in (0..bt).step_by(LOOP_UNROLL) {
        for o in 0..oc {
            // Initialize results array with bias
            let mut result = [0.0f32; LOOP_UNROLL];
            for ibt in 0..LOOP_UNROLL {
                result[ibt] = bias[o];
            }

            // Perform the main matrix multiplication with tiling
            for i in 0..c {
                let w = weight[i + o * c];
                for ibt in 0..LOOP_UNROLL {
                    let bt_idx = obt + ibt;
                    result[ibt] += inp[bt_idx * c + i] * w;
                }
            }

            // Write back results to the output buffer
            for ibt in 0..LOOP_UNROLL {
                let bt_idx = obt + ibt;
                out[bt_idx * oc + o] = result[ibt];
            }
        }
    }
}


fn test_matrix_mult_naive_backward(){
    let b = 4; // Batch size
    let t = 64; // Time steps
    let c = 768; // Input channels
    let oc = 768; // Output channels

    // Example inputs
    let inp = vec![0.1; b * t * c];
    let weight = vec![0.2; oc * c];
    let mut dweight = vec![0.2; oc * c];
    let bias = vec![0.3; oc];
    let mut dbias = vec![0.3; oc];
    let mut out = vec![0.0; b * t * oc];
    let mut dout = vec![1.0; b * t * oc];

    // Call the function
    // matmul_forward_naive(&mut out, &inp, &weight, &bias, b, t, c, oc);
    matmul_naive_backward(&mut out, &mut dout, &inp, &weight, &mut dweight, &bias, &mut dbias, b, t, c, oc);

    // println!("Output: {:?}", out);
    // println!("dWeight: {:?}", dweight);
}

fn test_matrix_mult_naive_forward(){
    let b = 4; // Batch size
    let t = 64; // Time steps
    let c = 768; // Input channels
    let oc = 768; // Output channels

    // Example inputs
    let inp = vec![0.1; b * t * c];
    let weight = vec![0.2; oc * c];
    let bias = vec![0.3; oc];
    let mut out = vec![0.0; b * t * oc];

    // Call the function
    // matmul_forward_naive(&mut out, &inp, &weight, &bias, b, t, c, oc);
    matmul_forward_naive(&mut out, &inp, &weight, &bias, b, t, c, oc);
    // println!("Output: {:?}", out);
    // println!("dWeight: {:?}", dweight);
}


fn test_matrix_mult_forward(){
    let b = 4; // Batch size
    let t = 64; // Time steps
    let c = 768; // Input channels
    let oc = 768; // Output channels

    // Example inputs
    let inp = vec![0.1; b * t * c];
    let weight = vec![0.2; oc * c];
    let bias = vec![0.3; oc];
    let mut out = vec![0.0; b * t * oc];

    // Call the function
    // matmul_forward_naive(&mut out, &inp, &weight, &bias, b, t, c, oc);
    matmul_forward(&mut out, &inp, &weight, &bias, b, t, c, oc);
    // println!("Output: {:?}", out);
    // println!("dWeight: {:?}", dweight);
}

fn test_matrix_mult_backward(){
    let b = 4; // Batch size
    let t = 64; // Time steps
    let c = 768; // Input channels
    let oc = 768; // Output channels

    // Example inputs
    let inp = vec![0.1; b * t * c];
    let weight = vec![0.2; oc * c];
    let mut dweight = vec![0.2; oc * c];
    let bias = vec![0.3; oc];
    let mut dbias = vec![0.3; oc];
    let mut out = vec![0.0; b * t * oc];
    let mut dout = vec![1.0; b * t * oc];

    // Call the function
    // matmul_forward_naive(&mut out, &inp, &weight, &bias, b, t, c, oc);
    matmul_backward(&mut out, &mut dout, &inp, &weight, &mut dweight, &bias, &mut dbias, b, t, c, oc);

    // println!("Output: {:?}", out);
    // println!("dWeight: {:?}", dweight);
}




#[autodiff(encoder_backward, Reverse, Duplicated, Const, Duplicated, Duplicated, Const, Const, Const)]
fn encoder_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    for b_idx in 0..b {
        for t_idx in 0..t {
            // Seek to the output position in out[b,t,:]
            let out_bt_start = b_idx * t * c + t_idx * c;
            let out_bt = &mut out[out_bt_start..out_bt_start + c];

            // Get the index of the token at inp[b, t]
            let ix = inp[b_idx * t + t_idx] as usize;

            // Seek to the position in wte corresponding to the token
            let wte_ix_start = ix * c;
            let wte_ix = &wte[wte_ix_start..wte_ix_start + c];

            // Seek to the position in wpe corresponding to the position
            let wpe_t_start = t_idx * c;
            let wpe_t = &wpe[wpe_t_start..wpe_t_start + c];

            // Add the two vectors and store the result in out[b,t,:]
            for i in 0..c {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}





#[autodiff(layernorm_backward, Reverse, Duplicated, Const, Const, Duplicated, Duplicated, Duplicated, Const, Const, Const)]
fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    let eps = 1e-5f32;
    for b_idx in 0..b {
        for t_idx in 0..t {
            let x_start = b_idx * t * c + t_idx * c;
            let x = &inp[x_start..x_start + c];

            // Calculate mean
            let m: f32 = x.iter().sum::<f32>() / c as f32;

            // Calculate variance
            let v: f32 = x.iter().map(|&xi| (xi - m).powi(2)).sum::<f32>() / c as f32;

            // Calculate reciprocal standard deviation
            let s = 1.0 / (v + eps).sqrt();

            // Seek to the output position in out[b,t,:]
            let out_bt_start = b_idx * t * c + t_idx * c;
            let out_bt = &mut out[out_bt_start..out_bt_start + c];

            for i in 0..c {
                let n = s * (x[i] - m); // Normalize
                out_bt[i] = n * weight[i] + bias[i]; // Scale and shift
            }

            // Cache the mean and rstd for the backward pass later
            mean[b_idx * t + t_idx] = m;
            rstd[b_idx * t + t_idx] = s;
        }
    }
}



#[autodiff(attention_backward, Reverse, Duplicated, Duplicated, Duplicated, Duplicated, Const, Const, Const, Const)]
fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
) {
    let c3 = c * 3;
    let hs = c / nh; // head size
    let scale = 1.0 / (hs as f32).sqrt();

    for b_idx in 0..b {
        for t_idx in 0..t {
            for h_idx in 0..nh {
                let query_t = &inp[b_idx * t * c3 + t_idx * c3 + h_idx * hs..];
                let preatt_bth = &mut preatt[b_idx * nh * t * t + h_idx * t * t + t_idx * t..];
                let att_bth = &mut att[b_idx * nh * t * t + h_idx * t * t + t_idx * t..];

                // Pass 1: Calculate query dot key and maxval
                let mut maxval = -10000.0f32; // TODO: something better
                for t2 in 0..=t_idx {
                    let key_t2 = &inp[b_idx * t * c3 + t2 * c3 + h_idx * hs + c..];

                    // (query_t) dot (key_t2)
                    let mut val = 0.0f32;
                    for i in 0..hs {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // Pass 2: Calculate the exp and keep track of sum
                let mut expsum = 0.0f32;
                for t2 in 0..=t_idx {
                    let expv = (preatt_bth[t2] - maxval).exp();
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // Pass 3: Normalize to get the softmax
                for t2 in 0..t {
                    if t2 <= t_idx {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask
                        att_bth[t2] = 0.0;
                    }
                }

                // Pass 4: Accumulate weighted values into the output of attention
                let out_bth = &mut out[b_idx * t * c + t_idx * c + h_idx * hs..];
                out_bth.iter_mut().take(hs).for_each(|o| *o = 0.0);
                for t2 in 0..=t_idx {
                    let value_t2 = &inp[b_idx * t * c3 + t2 * c3 + h_idx * hs + c * 2..];
                    let att_btht2 = att_bth[t2];
                    for i in 0..hs {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

fn test_attention_forward() {
    let b = 4;  // Batch size
    let t = 64; // Sequence length
    let c = 768; // Embedding size
    let nh = 12; // Number of heads

    // Input (B, T, 3C) with query, key, and value vectors
    let inp: Vec<f32> = (0..(b * t * 3 * c))
        .map(|i| (i as f32 + 1.0) / (b * t * 3 * c) as f32)
        .collect();

    // Allocate space for the outputs and intermediate states
    let mut out = vec![0.0; b * t * c];       // Output (B, T, C)
    let mut preatt = vec![0.0; b * nh * t * t]; // Pre-attention scores (B, NH, T, T)
    let mut att = vec![0.0; b * nh * t * t];    // Post-attention scores (B, NH, T, T)

    // Call the attention_forward function
    attention_forward(&mut out, &mut preatt, &mut att, &inp, b, t, c, nh);
}


fn test_attention_backward() {
    let b = 4;  // Batch size
    let t = 64; // Sequence length
    let c = 768; // Embedding size
    let nh = 12; // Number of heads

    // Input (B, T, 3C) with query, key, and value vectors
    let inp: Vec<f32> = (0..(b * t * 3 * c))
        .map(|i| (i as f32 + 1.0) / (b * t * 3 * c) as f32)
        .collect();

    let mut dinp = vec![0.0; b * t * 3 * c];

    // Allocate space for the outputs and intermediate states
    let mut out = vec![0.0; b * t * c];       // Output (B, T, C)
    let mut dout = vec![1.0_f32; b * t * c];
    let mut preatt = vec![0.0; b * nh * t * t]; // Pre-attention scores (B, NH, T, T)
    let mut dpreatt = vec![0.0; b * nh * t * t];
    let mut att = vec![0.0; b * nh * t * t];    // Post-attention scores (B, NH, T, T)
    let mut datt =  vec![0.0; b * nh * t * t];
    // Call the attention_forward function
    attention_backward(&mut out, &mut dout, &mut preatt, &mut dpreatt, &mut att, &mut datt, &inp, &mut dinp, b, t, c, nh);
}

fn test_layernorm_forward(){
    let b = 4; // Small size for B
    let t = 64; // Small size for T
    let c = 768; // Small size for C
    let inp: Vec<f32> = (0..(b * t * c))
        .map(|i| (i as f32 + 1.0) / (b * t * c) as f32)
        .collect();
    
    let mut out = vec![0.0; b * t * c];
    let mut mean = vec![0.0; b * t];
    let mut rstd = vec![0.0; b * t];
    let weight = vec![1.0; c]; // Scale weights as needed
    let bias = vec![0.0; c];  // Shift biases as needed

    layernorm_forward(&mut out, &mut mean, &mut rstd, &inp, &weight, &bias, b, t, c);
}

fn test_layernorm_backward() {
    let b = 4; // Small size for B
    let t = 64; // Small size for T
    let c = 768; // Small size for C

    let inp: Vec<f32> = (0..(b * t * c))
        .map(|i| (i as f32 + 1.0) / (b * t * c) as f32)
        .collect();

    let mut dinp = vec![0.0_f32; b * t * c];

    let mut out = vec![0.0_f32; b * t * c];
    let mut mean = vec![0.0_f32; b * t];
    let mut rstd = vec![0.0_f32; b * t];
    let weight = vec![1.0_f32; c]; // Scale weights as needed
    let mut dweight = vec![0.0_f32; c]; // Ensure f32 type
    let bias = vec![0.2_f32; c];  // Shift biases as needed
    let mut dbias = vec![0.0_f32; c];  // Ensure f32 type
    let mut dout = vec![1.0_f32; b * t * c];

    // Pass slices to match the function signature
    layernorm_backward(
        &mut out,
        &mut dout,
        &mut mean,
        &mut rstd,
        &inp,
        &mut dinp,
        &weight.as_slice(),
        &mut dweight.as_mut_slice(),
        &bias.as_slice(),
        &mut dbias.as_mut_slice(),
        b,
        t,
        c,
    );
}

// fn test_layer_norm(){
//     let b = 2; // Batch size
//     let t = 3; // Sequence length
//     let c = 4; // Embedding size

//     let inp = vec![
//         1.0, 2.0, 3.0, 4.0, // (b=0, t=0)
//         5.0, 6.0, 7.0, 8.0, // (b=0, t=1)
//         9.0, 10.0, 11.0, 12.0, // (b=0, t=2)
//         1.0, 3.0, 5.0, 7.0, // (b=1, t=0)
//         2.0, 4.0, 6.0, 8.0, // (b=1, t=1)
//         3.0, 5.0, 7.0, 9.0, // (b=1, t=2)
//     ];
//     let weight = vec![0.1, 0.2, 0.3, 0.4];
//     let mut dweight = vec![0; weight.len()];
//     let bias = vec![0.5, 0.6, 0.7, 0.8];
//     let mut dbias = vec![0; bias.len()];
//     let mut out = vec![0.0; b * t * c];
//     let mut mean = vec![0.0; b * t];
//     let mut rstd = vec![0.0; b * t];

//     layernorm_forward(&mut out, &mut mean, &mut rstd, &inp, &weight, &bias, b, t, c);
//     println!("Output: {:?}", out);

//     layernorm_backward(&mut out, )
// }


fn test_encoder_backward(){
    // Dimensions
    let b = 2; // Batch size
    let t = 3; // Sequence length
    let c = 4; // Embedding size

    // Inputs
    let mut dout = vec![1.0; b * t * c];
    let inp = vec![1, 2, 3, 0, 1, 2]; // Token IDs (B, T)
    let wte = vec![
        // Token embeddings (V, C) where V >= max(inp)
        0.1, 0.2, 0.3, 0.4, // Token 0
        0.5, 0.6, 0.7, 0.8, // Token 1
        0.9, 1.0, 1.1, 1.2, // Token 2
        1.3, 1.4, 1.5, 1.6, // Token 3
    ];
    let wpe = vec![
        // Positional embeddings (maxT, C)
        0.01, 0.02, 0.03, 0.04, // Position 0
        0.05, 0.06, 0.07, 0.08, // Position 1
        0.09, 0.10, 0.11, 0.12, // Position 2
    ];
    let mut out = vec![1.0; b * t * c]; // Output (B, T, C)
    let mut dwte = vec![0.0; wte.len()]; // Gradient of token embeddings (same shape as wte)
    let mut dwpe = vec![0.0; wpe.len()]; // Gradient of position embeddings (same shape as wpe)

    encoder_forward(&mut out, &inp, &wte, &wpe, b, t, c);

    println!("encoder forward: {:?}", out);

    encoder_backward(
        &mut out,
        &mut dout,
        &inp,
        &wte,
        &mut dwte,
        &wpe,
        &mut dwpe,
        b,
        t,
        c,
    );

    println!("dweight: {:?}", dwte);
}

// SOFTMAX

#[autodiff(crossentropy_backward, Reverse, Duplicated, Duplicated, Const, Const, Const, Const)]
fn crossentropy_forward(losses: &mut [f32], probs: &[f32], targets: &[i32], B: usize, T: usize, Vp: usize) {
    for b in 0..B {
        for t in 0..T {
            // loss = -log(probs[target])
            let probs_bt = &probs[b * T * Vp + t * Vp..];
            let ix = targets[b * T + t] as usize;
            losses[b * T + t] = -probs_bt[ix].ln();
        }
    }
}


fn test_crossentropy_forward() {
    // Example dimensions
    let B = 4;
    let T = 64;
    let Vp = 50304;
    let mut losses: Vec<f32> = vec![0.1; B * T * Vp];
    let probs: Vec<f32> = vec![0.5; B * T * Vp];
    let targets: Vec<i32> = vec![1; B * T];
    crossentropy_forward(&mut losses, &probs, &targets, B, T, Vp);
}




fn test_crossentropy_backward() {
    // Example dimensions
    let B = 4;
    let T = 64;
    let Vp = 50304;
    let mut losses: Vec<f32> = vec![0.1; B * T * Vp];
    let mut dlosses: Vec<f32> = vec![1.0; B * T * Vp];
    let mut probs: Vec<f32> = vec![0.5; B * T * Vp];
    let mut dprobs: Vec<f32> = vec![0.0; B * T * Vp];
    let targets: Vec<i32> = vec![1; B * T];
    crossentropy_backward(&mut dlosses, &mut losses, &mut probs, &mut dprobs, &targets, B, T, Vp);
}





// Function 1: Example function to benchmark

// Benchmarking function

fn benchmark_matmult(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison");

    // Benchmark function_one for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Enzyme Backward", "Matrix Mult"), |b| {
        b.iter(|| test_matrix_mult_backward());
    });

    // Benchmark function_two for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Forward", "Matrix Mult"), |b| {
        b.iter(|| test_matrix_mult_forward());
    });

    group.finish();
}



fn benchmark_layernorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison");

    // Benchmark function_one for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Enzyme Backward", "Layer Norm"), |b| {
        b.iter(|| test_layernorm_backward());
    });

    // Benchmark function_two for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Forward", "Layer Norm"), |b| {
        b.iter(|| test_layernorm_forward());
    });

    group.finish();
}

fn benchmark_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison");

    // Benchmark function_one for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Enzyme Backward", "Attention"), |b| {
        b.iter(|| test_attention_backward());
    });

    // Benchmark function_two for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Forward", "Attention"), |b| {
        b.iter(|| test_attention_forward());
    });

    group.finish();
}


fn benchmark_crossentropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison");

    // Benchmark function_one for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Enzyme Backward", "Cross Entropy"), |b| {
        b.iter(|| test_crossentropy_backward());
    });

    // Benchmark function_two for a fixed number of iterations
    group.bench_function(BenchmarkId::new("Forward", "Cross Entropy"), |b| {
        b.iter(|| test_crossentropy_forward());
    });

    group.finish();
}


criterion_group!(benches, benchmark_crossentropy);
criterion_main!(benches);
// fn main() {
