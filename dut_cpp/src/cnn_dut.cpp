
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../include/cnn_dut.h"

// this file implements the CNN_DUT class defined in cnn_dut.h
// Constructor
// test for checker.
CNN_DUT::CNN_DUT(
    const Tensor4D_8& conv1_w, const Vector32& conv1_b,
    const Tensor4D_8& conv2_w, const Vector32& conv2_b,
    const std::vector<std::vector<int8_t>>& dense1_w, const Vector32& dense1_b,
    const std::vector<std::vector<int8_t>>& dense2_w, const Vector32& dense2_b)
    : conv1_weights(conv1_w), conv1_bias(conv1_b),
      conv2_weights(conv2_w), conv2_bias(conv2_b),
      dense1_weights(dense1_w), dense1_bias(dense1_b),
      dense2_weights(dense2_w), dense2_bias(dense2_b) {}

// Hàm thực thi pipeline của CNN
std::vector<float> CNN_DUT::predict(const Tensor3D_8& image) { // predict nhận ảnh đầu vào 8-bit
    Tensor3D_32 conv1_out = conv2d(image, conv1_weights, conv1_bias);
    Tensor3D_8 relu1_out = relu(conv1_out);

    Tensor3D_32 conv2_out = conv2d(relu1_out, conv2_weights, conv2_bias);
    Tensor3D_8 relu2_out = relu(conv2_out);

    Tensor3D_8 pool_out = max_pooling2d(relu2_out, 3, 3);
    
    Vector8 flattened_out = flatten(pool_out);

    Vector32 dense1_out = fully_connected(flattened_out, dense1_weights, dense1_bias);
    Vector8 sigmoid_out = sigmoid(dense1_out);

    Vector32 dense2_out = fully_connected(sigmoid_out, dense2_weights, dense2_bias);
    //trả về softmax của dense2_out
    return softmax(dense2_out);
}


// --- deploy the details---

Tensor3D_32 CNN_DUT::conv2d(const Tensor3D_8& input, const Tensor4D_8& kernel, const Vector32& bias) {
    size_t in_channels = input.size();
    size_t in_height = input[0].size();
    size_t in_width = input[0][0].size();

    size_t out_channels = kernel.size();
    size_t kernel_height = kernel[0][0].size();
    size_t kernel_width = kernel[0][0][0].size();

    size_t out_height = in_height - kernel_height + 1;
    size_t out_width = in_width - kernel_width + 1;

    Tensor3D_32 output(out_channels, std::vector<std::vector<int32_t>>(out_height, std::vector<int32_t>(out_width, 0)));

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t oh = 0; oh < out_height; ++oh) {
            for (size_t ow = 0; ow < out_width; ++ow) {
                // *** LỖI CỐ Ý #1: Tràn số trong tích chập ***
                // Tương tự như MLP, dùng int16_t có thể gây tràn số khi tích lũy.
                int16_t acc = 0; 
                for (size_t ic = 0; ic < in_channels; ++ic) {
                    for (size_t kh = 0; kh < kernel_height; ++kh) {
                        for (size_t kw = 0; kw < kernel_width; ++kw) {
                            acc += static_cast<int16_t>(input[ic][oh + kh][ow + kw]) * static_cast<int16_t>(kernel[oc][ic][kh][kw]);
                        }
                    }
                }
                output[oc][oh][ow] = static_cast<int32_t>(acc) + bias[oc];
            }
        }
    }
    return output;
}

Tensor3D_8 CNN_DUT::relu(const Tensor3D_32& input) {
    Tensor3D_8 output(input.size(), std::vector<std::vector<int8_t>>(input[0].size(), std::vector<int8_t>(input[0][0].size())));
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[0].size(); ++j) {
            for (size_t k = 0; k < input[0][0].size(); ++k) {
                int32_t val = std::max(0, input[i][j][k]);
                 // *** LỖI CỐ Ý #2: Lỗi cắt bỏ (clipping) sai ***
                if (val > 125) val = 125; // Sai! Phải là 127
                output[i][j][k] = static_cast<int8_t>(val);
            }
        }
    }
    return output;
}

Tensor3D_8 CNN_DUT::max_pooling2d(const Tensor3D_8& input, int pool_size, int stride) {
    size_t in_channels = input.size();
    size_t in_height = input[0].size();
    size_t in_width = input[0][0].size();

    size_t out_height = in_height / stride;
    size_t out_width = in_width / stride;
    
    // *** LỖI CỐ Ý #3: Sai logic tính toán kích thước đầu ra ***
    if (in_height % stride != 0) out_height++; // Dòng này có thể sai hoặc đúng tùy theo cách làm tròn
    // Tôi sẽ bỏ qua việc tính toán làm tròn đúng để tạo ra lỗi.

    Tensor3D_8 output(in_channels, std::vector<std::vector<int8_t>>(out_height, std::vector<int8_t>(out_width, 0)));

    for (size_t c = 0; c < in_channels; ++c) {
        for (size_t oh = 0; oh < out_height; ++oh) {
            for (size_t ow = 0; ow < out_width; ++ow) {
                int8_t max_val = std::numeric_limits<int8_t>::min();
                for (int ph = 0; ph < pool_size; ++ph) {
                    for (int pw = 0; pw < pool_size; ++pw) {
                        size_t ih = oh * stride + ph;
                        size_t iw = ow * stride + pw;
                        if (ih < in_height && iw < in_width) {
                            if (input[c][ih][iw] > max_val) {
                                max_val = input[c][ih][iw];
                            }
                        }
                    }
                }
                output[c][oh][ow] = max_val;
            }
        }
    }
    return output;
}


Vector8 CNN_DUT::flatten(const Tensor3D_8& input) {
    Vector8 output;
    output.reserve(input.size() * input[0].size() * input[0][0].size());
    // Keras flatten theo channel -> height -> width
    for (size_t i = 0; i < input[0].size(); ++i) {
        for (size_t j = 0; j < input[0][0].size(); ++j) {
            for (size_t k = 0; k < input.size(); ++k) {
                output.push_back(input[k][i][j]);
            }
        }
    }
    return output;
}
// Fully connected layer
Vector32 CNN_DUT::fully_connected(const Vector8& input, const std::vector<std::vector<int8_t>>& weights, const Vector32& bias) {
    Vector32 output(weights[0].size());
    for (size_t j = 0; j < weights[0].size(); ++j) {
        int32_t acc = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            acc += static_cast<int32_t>(input[i]) * static_cast<int32_t>(weights[i][j]);
        }
        output[j] = acc + bias[j];
    }
    return output;
}
// Sigmoid activation function
Vector8 CNN_DUT::sigmoid(const Vector32& input) {
    Vector8 output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        // Mô phỏng LUT (Look-up table) bằng phép tính float
        float val_float = 1.0f / (1.0f + std::exp(-static_cast<float>(input[i]) / 32.0f)); // Chia cho một scale factor
        // Lượng tử hóa kết quả về int8
        output[i] = static_cast<int8_t>(val_float * 127);
    }
    return output;
}

// Softmax function
std::vector<float> CNN_DUT::softmax(const Vector32& input) {
    std::vector<float> output(input.size());
    float max_val = -std::numeric_limits<float>::max();
    for (int32_t val : input) {
        if (static_cast<float>(val) > max_val) {
            max_val = static_cast<float>(val);
        }
    }
    float sum_exp = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float exp_val = std::exp((static_cast<float>(input[i]) - max_val) / 64.0f); // Thêm scale factor
        output[i] = exp_val;
        sum_exp += exp_val;
    }
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum_exp;
    }
    return output;
}