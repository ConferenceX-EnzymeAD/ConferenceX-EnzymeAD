import pandas as pd
import matplotlib.pyplot as plt

# Data for the benchmarks
benchmarks = {
    "matmul_naive": {
        "no_omp": {
            "naive_scaled": 147324851,
            "enzyme_scaled": 172364388,
        },
    },
    "matmul": {
        "no_omp": {
            "naive_scaled": 34753332,
            "enzyme_scaled": 59365196,
        },
    },
    "attention_forward": {
        "no_omp": {
            "forward_scaled": 11236163,
            "enzyme_scaled": 21002606,
        },
    },
    "cross_entropy": {
        "no_omp": {
            "forward_scaled": 3265,
            "enzyme_scaled": 3795,
        },
    },
}

data = {
    "Benchmark": ["MatMul_Naive", "MatMul", "Attention Forward", "Cross Entropy"],
    "No Enzyme": [
        benchmarks["matmul_naive"]["no_omp"]["naive_scaled"],
        benchmarks["matmul"]["no_omp"]["naive_scaled"],
        benchmarks["attention_forward"]["no_omp"]["forward_scaled"],
        benchmarks["cross_entropy"]["no_omp"]["forward_scaled"],
    ],
    "Enzyme": [
        benchmarks["matmul_naive"]["no_omp"]["enzyme_scaled"],
        benchmarks["matmul"]["no_omp"]["enzyme_scaled"],
        benchmarks["attention_forward"]["no_omp"]["enzyme_scaled"],
        benchmarks["cross_entropy"]["no_omp"]["enzyme_scaled"],
    ],
}

df = pd.DataFrame(data)


plt.figure(figsize=(12, 6))
bar_width = 0.35
x = range(len(df))

plt.bar(x, df["No Enzyme"], width=bar_width, label="No Enzyme")
plt.bar([p + bar_width for p in x], df["Enzyme"], width=bar_width, label="Enzyme")

plt.xticks([p + bar_width / 2 for p in x], df["Benchmark"])
plt.yscale('log')
plt.ylabel("Time (ns) [Log Scale]")
plt.title("Performance Comparison: No Enzyme vs Enzyme (No OMP) [Log Scale]")
plt.legend()

plt.tight_layout()
plt.show()