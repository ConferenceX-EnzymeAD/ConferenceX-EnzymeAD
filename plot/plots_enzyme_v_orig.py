import matplotlib.pyplot as plt


### PLOT FOR TRAINING AND VLAIDATION LOSS FOR ORIGINAL GPT

steps_train = list(range(41))
train_losses = [
    4.677772, 5.191522, 4.438629, 4.138455, 4.144238, 3.834684, 4.298056,
    4.280747, 4.249753, 4.391604, 3.912615, 3.737814, 3.840917, 4.367947,
    4.130485, 4.012578, 3.796071, 4.355925, 3.766852, 4.552072, 4.527331,
    4.065798, 3.965314, 3.449410, 4.490952, 4.035361, 3.445302, 3.993789,
    4.199468, 4.538459, 4.306293, 4.851405, 4.577482, 4.124942, 4.330319,
    3.399417, 3.661206, 3.330452, 3.567852, 3.902004, 3.952986
]
steps_val = list(range(0, 41, 5))
val_losses = [
    5.325521, 4.513729, 4.416496, 4.355394, 4.329332, 4.311923, 4.300247,
    4.303848, 4.291716
]

plt.figure(figsize=(12, 6))
plt.plot(steps_train, train_losses, label="Training Loss", marker='o', linestyle='-', alpha=0.7)
plt.plot(steps_val, val_losses, label="Validation Loss", marker='s', linestyle='--', color='orange', alpha=0.8)
plt.title("Training and Validation Loss For Original GPT")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# ===============================================

## TRAINING AND VALIDATION LOSS FOR ENZYME GPT
steps_train = list(range(41))
train_losses = [
    4.677772, 5.398969, 4.989898, 4.645216, 4.535881, 4.567311, 4.434838,
    4.434444, 4.517156, 4.570694, 4.472080, 4.435953, 4.210717, 4.682723,
    4.537892, 4.591377, 4.118345, 4.507001, 4.214896, 4.830546, 4.526665,
    4.420367, 4.184669, 3.734571, 4.594570, 4.222163, 3.847980, 4.220455,
    4.455548, 4.532023, 4.316846, 4.917089, 4.516826, 4.123207, 4.465712,
    3.622287, 3.876152, 3.549468, 3.562306, 4.044739, 4.081340
]
steps_val = list(range(0, 41, 5))
val_losses = [
    5.325521, 5.010791, 4.795465, 4.661596, 4.546501, 4.452114, 4.389969,
    4.360292, 4.326075
]

plt.figure(figsize=(12, 6))
plt.plot(steps_train, train_losses, label="Training Loss", marker='o', linestyle='-', alpha=0.7)
plt.plot(steps_val, val_losses, label="Validation Loss", marker='s', linestyle='--', color='orange', alpha=0.8)
plt.title("Training and Validation Loss For Enzyme GPT")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# ===============================================
## PLOTS FOR ENZYME VS ORIGINAL BENCHMARKS 


categories = ["Forward Pass", "Backward Pass"]


no_omp_enzyme_25 = [7764991010, 70440000000]  # Enzyme no OMP
no_omp_original_25 = [7686845422, 19910000000]  # Original no OMP
omp_enzyme_25 = [877361341, 47049000000]  # Enzyme with OMP
omp_original_25 = [895471955, 2404048634]  # Original with OMP

# Plot 1: Enzyme vs Original (No OMP, 25 iterations)
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(categories))
bar_width = 0.35

ax1.bar(x - bar_width / 2, no_omp_enzyme_25, bar_width, label="Enzyme", color='blue')
ax1.bar(x + bar_width / 2, no_omp_original_25, bar_width, label="Original", color='orange')

ax1.set_xlabel("Benchmark Category")
ax1.set_ylabel("Execution Time (ns)")
ax1.set_title("Enzyme vs Original (No OMP)")
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()

# Plot 2: Enzyme vs Original (With OMP, 25 iterations)
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(x - bar_width / 2, omp_enzyme_25, bar_width, label="Enzyme", color='blue')
ax2.bar(x + bar_width / 2, omp_original_25, bar_width, label="Original", color='orange')

ax2.set_xlabel("Benchmark Category")
ax2.set_ylabel("Execution Time (ns)")
ax2.set_title("Enzyme vs Original (With OMP)")
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()

# Plot 3: Original (No OMP vs With OMP, 25 iterations)
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.bar(x - bar_width / 2, no_omp_original_25, bar_width, label="No OMP", color='blue')
ax3.bar(x + bar_width / 2, omp_original_25, bar_width, label="With OMP", color='orange')

ax3.set_xlabel("Benchmark Category")
ax3.set_ylabel("Execution Time (ns)")
ax3.set_title("Original: No OMP vs With OMP ")
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()

# Plot 4: Enzyme (No OMP vs With OMP, 25 iterations)
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.bar(x - bar_width / 2, no_omp_enzyme_25, bar_width, label="No OMP", color='blue')
ax4.bar(x + bar_width / 2, omp_enzyme_25, bar_width, label="With OMP", color='orange')

ax4.set_xlabel("Benchmark Category")
ax4.set_ylabel("Execution Time (ns)")
ax4.set_title("Enzyme: No OMP vs With OMP")
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()

plt.tight_layout()
plt.show()


# below is to combine graphs into one plot

fig, ax = plt.subplots(figsize=(16, 10))
x = np.arange(len(categories))
bar_width = 0.15

ax.bar(x - 1.5 * bar_width, no_omp_enzyme_25, bar_width, label="Enzyme (No OMP)", color='blue')
ax.bar(x - 0.5 * bar_width, no_omp_original_25, bar_width, label="Original (No OMP)", color='orange')
ax.bar(x + 0.5 * bar_width, omp_enzyme_25, bar_width, label="Enzyme (With OMP)", color='green')
ax.bar(x + 1.5 * bar_width, omp_original_25, bar_width, label="Original (With OMP)", color='red')

ax.set_xlabel("Benchmark Category", fontsize=14)
ax.set_ylabel("Execution Time (ns)", fontsize=14)
ax.set_title("Combined Benchmarks: Enzyme vs Original, With and Without OMP (25 iterations)", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Enzyme vs Original (No OMP, 25 iterations)
axs[0, 0].bar(x - bar_width / 2, no_omp_enzyme_25, bar_width, label="Enzyme", color='blue')
axs[0, 0].bar(x + bar_width / 2, no_omp_original_25, bar_width, label="Original", color='orange')
axs[0, 0].set_title("Enzyme vs Original (No OMP, 25 iterations)", fontsize=14)
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(categories, fontsize=12)
axs[0, 0].set_ylabel("Execution Time (ns)", fontsize=12)
axs[0, 0].legend(fontsize=10)

# Plot 2: Enzyme vs Original (With OMP, 25 iterations)
axs[0, 1].bar(x - bar_width / 2, omp_enzyme_25, bar_width, label="Enzyme", color='green')
axs[0, 1].bar(x + bar_width / 2, omp_original_25, bar_width, label="Original", color='red')
axs[0, 1].set_title("Enzyme vs Original (With OMP, 25 iterations)", fontsize=14)
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(categories, fontsize=12)
axs[0, 1].legend(fontsize=10)

# Plot 3: Original (No OMP vs With OMP, 25 iterations)
axs[1, 0].bar(x - bar_width / 2, no_omp_original_25, bar_width, label="No OMP", color='purple')
axs[1, 0].bar(x + bar_width / 2, omp_original_25, bar_width, label="With OMP", color='pink')
axs[1, 0].set_title("Original: No OMP vs With OMP (25 iterations)", fontsize=14)
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(categories, fontsize=12)
axs[1, 0].set_ylabel("Execution Time (ns)", fontsize=12)
axs[1, 0].legend(fontsize=10)

# Plot 4: Enzyme (No OMP vs With OMP, 25 iterations)
axs[1, 1].bar(x - bar_width / 2, no_omp_enzyme_25, bar_width, label="No OMP", color='cyan')
axs[1, 1].bar(x + bar_width / 2, omp_enzyme_25, bar_width, label="With OMP", color='magenta')
axs[1, 1].set_title("Enzyme: No OMP vs With OMP (25 iterations)", fontsize=14)
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(categories, fontsize=12)
axs[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.show()

