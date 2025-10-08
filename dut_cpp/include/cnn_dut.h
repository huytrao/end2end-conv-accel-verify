#ifndef CNN_DUT_H
#define CNN_DUT_H
// cnn_dut.h - Định nghĩa lớp CNN_DUT và các kiểu dữ liệu liên quan
#include <vector>
#include <cstdint>

// Định nghĩa các kiểu dữ liệu để làm việc với Tensor
// Vector 1D
using Vector8 = std::vector<int8_t>;
// Tensor 3D (ví dụ: ảnh [channels][height][width])
using Tensor3D_8 = std::vector<std::vector<std::vector<int8_t>>>;
// Tensor 4D (ví dụ: kernel Conv [out_ch][in_ch][ky][kx])
using Tensor4D_8 = std::vector<std::vector<std::vector<std::vector<int8_t>>>>;

// Kiểu dữ liệu trung gian cho tính toán
using Vector32 = std::vector<int32_t>;
using Tensor3D_32 = std::vector<std::vector<std::vector<int32_t>>>;


class CNN_DUT {
public:
    // Constructor để nạp các trọng số đã được lượng tử hóa
    CNN_DUT(
        const Tensor4D_8& conv1_w, const Vector32& conv1_b, // Trọng số Conv1 là vector 32-bit
        const Tensor4D_8& conv2_w, const Vector32& conv2_b, // Trọng số Conv2 là vector 32-bit
        const std::vector<std::vector<int8_t>>& dense1_w, const Vector32& dense1_b, // Trọng số Dense1 là int8_t
        const std::vector<std::vector<int8_t>>& dense2_w, const Vector32& dense2_b // Trọng số Dense2 là int8_t 
    );

    // Hàm thực thi chính
    std::vector<float> predict(const Tensor3D_8& image);

private:
    // save weight and bias
    Tensor4D_8 conv1_weights;
    Vector32 conv1_bias;
    Tensor4D_8 conv2_weights;
    Vector32 conv2_bias;
    std::vector<std::vector<int8_t>> dense1_weights;
    Vector32 dense1_bias;
    std::vector<std::vector<int8_t>> dense2_weights;
    Vector32 dense2_bias;

    // stimulate block of CNN
    Tensor3D_32 conv2d(const Tensor3D_8& input, const Tensor4D_8& kernel, const Vector32& bias);
    Tensor3D_8 relu(const Tensor3D_32& input);
    Tensor3D_8 max_pooling2d(const Tensor3D_8& input, int pool_size, int stride);
    Vector8 flatten(const Tensor3D_8& input);
    Vector32 fully_connected(const Vector8& input, const std::vector<std::vector<int8_t>>& weights, const Vector32& bias);
    Vector8 sigmoid(const Vector32& input); // Keras model dùng sigmoid
    std::vector<float> softmax(const Vector32& input);
};

#endif // CNN_DUT_H