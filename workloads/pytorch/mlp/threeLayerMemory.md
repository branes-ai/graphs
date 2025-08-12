# Memory requirements for a 3-layer MLP

A detailed comparison of **parameter and activation memory** for your 3-layer MLP across six different hidden dimensions: **1K, 2K, 4K, 8K, 16K, and 32K**.

---

## Memory Comparison Table

| Hidden Dim | Param Memory (fc1) | Param Memory (fc2) | Param Memory (fc3) | **Total Param Memory** | Activation Memory (fc1) | Activation Memory (fc2) | Activation Memory (fc3) | **Total Activation Memory** |
|------------|--------------------|---------------------|---------------------|-------------------------|--------------------------|--------------------------|--------------------------|------------------------------|
| **1K**     | 4.00 MB            | 4.00 MB             | 0.50 MB             | **8.50 MB**             | 40 KB                    | 40 KB                    | 5 KB                     | **85 KB**                   |
| **2K**     | 8.00 MB            | 16.00 MB            | 1.00 MB             | **25.00 MB**            | 80 KB                    | 80 KB                    | 5 KB                     | **165 KB**                  |
| **4K**     | 16.00 MB           | 64.00 MB            | 2.00 MB             | **82.00 MB**            | 160 KB                   | 160 KB                   | 5 KB                     | **325 KB**                  |
| **8K**     | 32.00 MB           | 256.00 MB           | 4.00 MB             | **292.00 MB**           | 320 KB                   | 320 KB                   | 5 KB                     | **645 KB**                  |
| **16K**    | 64.00 MB           | 1.00 GB             | 8.00 MB             | **1.072 GB**            | 640 KB                   | 640 KB                   | 5 KB                     | **1.25 MB**                 |
| **32K**    | 128.00 MB          | 4.00 GB             | 16.00 MB            | **4.144 GB**            | 1.25 MB                  | 1.25 MB                  | 5 KB                     | **2.5 MB**                  |

---

### How These Numbers Were Calculated

- **Param Memory**:
  - `fc1`: `input_dim × hidden_dim × 4 bytes + hidden_dim × 4 bytes`
  - `fc2`: `hidden_dim × hidden_dim × 4 bytes + hidden_dim × 4 bytes`
  - `fc3`: `hidden_dim × output_dim × 4 bytes + output_dim × 4 bytes`
- **Activation Memory**:
  - Each activation: `batch_size × hidden_dim × 4 bytes`
  - Output layer: `batch_size × output_dim × 4 bytes`

---

## Observations

- Memory scales **quadratically** in `fc2` due to the square weight matrix.
- Activation memory is relatively small compared to parameter memory.
- At `32K`, you're already pushing **4+ GB** of parameter memory—GPU caution advised!

Would you like to visualize this trend with a plot or explore architectural tweaks to reduce memory pressure?