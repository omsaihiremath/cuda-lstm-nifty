// CUDA + LibTorch based Monte Carlo + LSTM forecast code for Nifty50
// Includes Monte Carlo simulation on CPU/GPU, LSTM with validation loss, and output ranges

#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <torch/torch.h>

#define BLOCK_SIZE 256

__global__ void monteCarloKernel(float *results, float S0, float mu, float sigma, float dt, int numSteps, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    float S = S0;
    for (int t = 0; t < numSteps; t++) {
        float gauss = curand_normal(&state);
        S *= expf((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * gauss);
    }
    results[idx] = S;
}

void monteCarloCPU(float *results, int numSimulations, float S0, float mu, float sigma, float dt, int numSteps) {
    for (int i = 0; i < numSimulations; i++) {
        float S = S0;
        for (int t = 0; t < numSteps; t++) {
            float u1 = static_cast<float>(rand()) / RAND_MAX;
            float u2 = static_cast<float>(rand()) / RAND_MAX;
            float gauss = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            S *= expf((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * gauss);
        }
        results[i] = S;
    }
}

struct MultiOutputLSTMModelImpl : torch::nn::Module {
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear fc{nullptr};

    MultiOutputLSTMModelImpl(int input_size, int hidden_size, int output_horizon) {
        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).batch_first(true)));
        dropout = register_module("dropout", torch::nn::Dropout(0.2));
        fc = register_module("fc", torch::nn::Linear(hidden_size, output_horizon));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto lstm_out = std::get<0>(lstm->forward(x));
        auto last_step = lstm_out.select(1, lstm_out.size(1) - 1);
        return fc->forward(dropout->forward(last_step));
    }
};
TORCH_MODULE(MultiOutputLSTMModel);

class MyDataset : public torch::data::datasets::Dataset<MyDataset> {
    torch::Tensor data_, targets_;
public:
    MyDataset(torch::Tensor data, torch::Tensor targets) : data_(std::move(data)), targets_(std::move(targets)) {}

    torch::data::Example<> get(size_t index) override {
        return {data_[index], targets_[index]};
    }

    torch::optional<size_t> size() const override {
        return data_.size(0);
    }
};

std::vector<float> loadClosePrices(const std::string &filename) {
    std::vector<float> prices;
    std::ifstream file(filename);
    std::string line;
    for (int i = 0; i < 3; i++) std::getline(file, line); // skip headers
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string date, closeStr;
        if (!std::getline(ss, date, ',')) continue;
        if (!std::getline(ss, closeStr, ',')) continue;
        try { prices.push_back(std::stof(closeStr)); } catch (...) {}
    }
    return prices;
}

void normalizeData(std::vector<float> &data, float &minVal, float &maxVal) {
    minVal = *std::min_element(data.begin(), data.end());
    maxVal = *std::max_element(data.begin(), data.end());
    for (auto &val : data) val = (val - minVal) / (maxVal - minVal);
}

std::pair<torch::Tensor, torch::Tensor> createSequences(const std::vector<float> &data, int window, int horizon) {
    std::vector<std::vector<float>> X, Y;
    for (size_t i = 0; i + window + horizon <= data.size(); i++) {
        X.push_back({data.begin() + i, data.begin() + i + window});
        Y.push_back({data.begin() + i + window, data.begin() + i + window + horizon});
    }
    torch::Tensor inputs = torch::empty({(int)X.size(), window, 1});
    torch::Tensor targets = torch::empty({(int)Y.size(), horizon});
    for (size_t i = 0; i < X.size(); i++) {
        for (int j = 0; j < window; j++) inputs[i][j][0] = X[i][j];
        for (int h = 0; h < horizon; h++) targets[i][h] = Y[i][h];
    }
    return {inputs, targets};
}

float computeQuantile(std::vector<float> v, float q) {
    std::sort(v.begin(), v.end());
    float pos = q * (v.size() - 1);
    int idx = (int)pos;
    float frac = pos - idx;
    return (1 - frac) * v[idx] + frac * v[std::min(idx + 1, (int)v.size() - 1)];
}

