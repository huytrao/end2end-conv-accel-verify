import numpy as np

# Các hàm này phải mô phỏng chính xác logic của C++ (phiên bản không lỗi)
# để có một "đáp án chuẩn" thực sự.

def np_conv2d(input_tensor, weights, bias):
    in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weights.shape
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1
    
    output_tensor = np.zeros((out_channels, out_height, out_width), dtype=np.int32)

    for oc in range(out_channels):
        for oh in range(out_height):
            for ow in range(out_width):
                # Tích chập trên một cửa sổ
                receptive_field = input_tensor[:, oh:oh+kernel_height, ow:ow+kernel_width]
                # Chuyển đổi kernel và receptive field sang int32 để tránh tràn số khi nhân
                # Đây là phiên bản "đúng" mà C++ DUT nên có
                acc = np.sum(receptive_field.astype(np.int32) * weights[oc].astype(np.int32))
                output_tensor[oc, oh, ow] = acc + bias[oc]
                
    return output_tensor

def np_relu(input_tensor):
    # Lượng tử hóa lại về int8 sau ReLU
    output = np.maximum(0, input_tensor)
    output = np.clip(output, -128, 127) # Logic clipping đúng
    return output.astype(np.int8)

def np_max_pooling2d(input_tensor, pool_size=3, stride=3):
    in_channels, in_height, in_width = input_tensor.shape
    out_height = in_height // stride
    out_width = in_width // stride
    
    output_tensor = np.zeros((in_channels, out_height, out_width), dtype=np.int8)

    for c in range(in_channels):
        for oh in range(out_height):
            for ow in range(out_width):
                h_start, w_start = oh * stride, ow * stride
                h_end, w_end = h_start + pool_size, w_start + pool_size
                window = input_tensor[c, h_start:h_end, w_start:w_end]
                output_tensor[c, oh, ow] = np.max(window)
                
    return output_tensor

def np_flatten(input_tensor):
    # Keras flatten theo (height, width, channel)
    # Phải khớp với C++
    return input_tensor.transpose(1, 2, 0).flatten()

def np_fully_connected(input_vector, weights, bias):
    return (input_vector.astype(np.int32) @ weights.astype(np.int32)) + bias

def np_sigmoid(input_vector):
    # Mô phỏng logic lượng tử hóa tương tự C++
    float_vals = 1.0 / (1.0 + np.exp(-input_vector / 32.0))
    return (float_vals * 127).astype(np.int8)

def np_softmax(input_vector):
    # Mô phỏng logic tương tự C++
    input_float = input_vector.astype(np.float32) / 64.0
    input_float -= np.max(input_float)
    exps = np.exp(input_float)
    return exps / np.sum(exps)

class GoldenModel:
    def __init__(self, weights, biases):
        self.conv1_w, self.conv2_w, self.dense1_w, self.dense2_w = weights
        self.conv1_b, self.conv2_b, self.dense1_b, self.dense2_b = biases

    def predict(self, image_tensor):
        conv1_out = np_conv2d(image_tensor, self.conv1_w, self.conv1_b)
        relu1_out = np_relu(conv1_out)
        
        conv2_out = np_conv2d(relu1_out, self.conv2_w, self.conv2_b)
        relu2_out = np_relu(conv2_out)

        pool_out = np_max_pooling2d(relu2_out)
        
        flat_out = np_flatten(pool_out)

        dense1_out = np_fully_connected(flat_out, self.dense1_w, self.dense1_b)
        sigmoid_out = np_sigmoid(dense1_out)
        
        dense2_out = np_fully_connected(sigmoid_out, self.dense2_w, self.dense2_b)

        return np_softmax(dense2_out)