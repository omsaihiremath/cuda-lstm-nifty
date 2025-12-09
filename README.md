# 📈 CUDA Monte Carlo & LSTM Forecasting – Nifty 50

This project implements a dual forecasting approach on the Nifty 50 stock index using:
- **Monte Carlo Simulations** (on CPU & GPU)
- **LSTM Neural Networks** (using LibTorch on CPU & GPU)

It provides a side-by-side comparison of CPU and GPU execution, and is ideal for demonstrating financial modeling, time series forecasting, and accelerated computing.

---

## 🚀 Features

- 📊 Predict next **1 week** and **1 month** Nifty 50 prices
- ⚙️ **Monte Carlo simulation** via CUDA kernel
- 🤖 **Multi-output LSTM** implemented with LibTorch
- 🧠 Uses last 60-day price window for prediction
- 📈 Logs:
  - Prediction Ranges (10%–90% quantile range)
  - Ensemble forecast (Monte Carlo + LSTM)
  - **Validation loss per epoch** (LSTM)
  - **Yearly accuracy**, **percent error**, and **estimated returns**

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `project.cu` | Main CUDA + LibTorch implementation |
| `nifty50.csv` | CSV file with historical closing prices |
| `loss_log.txt` | Logs training and validation loss per epoch |
| `yearly_results.csv` | Summary of accuracy and return per year |
| `CMakeLists.txt` | Build configuration file |

---

## 🛠️ Requirements

- CUDA-enabled GPU
- LibTorch (C++ distribution of PyTorch)
- CMake
- NVIDIA Toolkit & Compiler

---

## 🧪 Build & Run Instructions

```bash
# Clone the repository
git clone https://github.com/omsaihiremath/cuda-lstm-nifty.git
cd cuda-lstm-nifty

# Make a build directory
mkdir build && cd build

# Set environment variables for LibTorch (change path accordingly)
export LIBTORCH=/path/to/libtorch
export CPATH=$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include:$CPATH
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Run CMake and build
cmake .. -DCMAKE_PREFIX_PATH=$LIBTORCH
make

# Copy input file and execute
cp ../nifty50.csv .
./MyProject
