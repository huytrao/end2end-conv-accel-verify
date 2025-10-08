import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os

# Thêm đường dẫn tới module C++ đã được biên dịch
# Thư mục build sẽ chứa file cnn_dut_module.so (hoặc .pyd trên Windows)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

sys.path.append('../build/CmakeFiles')
import cnn_dut_module
from golden_model import GoldenModel

# --- Hàm tiện ích cho lượng tử hóa ---

def quantize_tensor(tensor, scale_factor):
    """Lượng tử hóa một tensor float32 thành int8."""
    quantized = np.round(tensor * scale_factor)
    return np.clip(quantized, -128, 127).astype(np.int8)

def quantize_bias(tensor, input_scale, weight_scale):
    """Lượng tử hóa bias."""
    # Bias không nhân với input nên scale factor khác
    return (tensor * input_scale * weight_scale).astype(np.int32)

def get_quantized_params(model):
    """Trích xuất và lượng tử hóa trọng số từ mô hình Keras."""
    print("Extracting and quantizing weights...")
    
    # Định nghĩa các scale factor (trong thực tế sẽ được tính toán cẩn thận)
    input_scale = 127.0
    conv1_w_scale = 127.0 / np.max(np.abs(model.layers[1].get_weights()[0]))
    conv2_w_scale = 127.0 / np.max(np.abs(model.layers[2].get_weights()[0]))
    dense1_w_scale = 127.0 / np.max(np.abs(model.layers[6].get_weights()[0]))
    dense2_w_scale = 127.0 / np.max(np.abs(model.layers[7].get_weights()[0]))
    
    # Conv1
    conv1_w_f, conv1_b_f = model.layers[1].get_weights()
    conv1_w_q = quantize_tensor(conv1_w_f, conv1_w_scale).transpose(3, 2, 0, 1) # HWIO -> OCHW
    conv1_b_q = quantize_bias(conv1_b_f, input_scale, conv1_w_scale)

    # Conv2
    conv2_w_f, conv2_b_f = model.layers[2].get_weights()
    conv2_w_q = quantize_tensor(conv2_w_f, conv2_w_scale).transpose(3, 2, 0, 1)
    conv2_b_q = quantize_bias(conv2_b_f, 1.0, conv2_w_scale) # Giả sử scale của relu output là 1.0

    # Dense1
    dense1_w_f, dense1_b_f = model.layers[6].get_weights()
    dense1_w_q = quantize_tensor(dense1_w_f, dense1_w_scale)
    dense1_b_q = quantize_bias(dense1_b_f, 1.0, dense1_w_scale)

    # Dense2
    dense2_w_f, dense2_b_f = model.layers[7].get_weights()
    dense2_w_q = quantize_tensor(dense2_w_f, dense2_w_scale)
    dense2_b_q = quantize_bias(dense2_b_f, 1.0, dense2_w_scale)
    
    q_weights = [conv1_w_q, conv2_w_q, dense1_w_q, dense2_w_q]
    q_biases = [conv1_b_q.tolist(), conv2_b_q.tolist(), dense1_b_q.tolist(), dense2_b_q.tolist()]
    
    print("Quantization complete.")
    return q_weights, q_biases, input_scale


def main():
    # --- 1. Tải dữ liệu và huấn luyện mô hình Keras ---
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    img_rows, img_cols = 28, 28
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    model_path = "mnist_cnn.h5"
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model = keras.models.load_model(model_path)
    else:
        print("Training new model...")
        inpx = keras.layers.Input(shape=(img_rows, img_cols, 1))
        layer1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
        layer2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(layer1)
        layer3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(layer2) # Thêm stride để khớp logic
        layer4 = keras.layers.Dropout(0.5)(layer3)
        layer5 = keras.layers.Flatten()(layer4)
        layer6 = keras.layers.Dense(250, activation='sigmoid')(layer5)
        layer7 = keras.layers.Dense(10, activation='softmax')(layer6)
        model = keras.models.Model(inpx, layer7)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train_cat, epochs=5, batch_size=128, validation_split=0.1)
        model.save(model_path)

    score = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Keras float32 model accuracy: {score[1]*100:.2f}%")
    
    # --- 2. Trích xuất và lượng tử hóa trọng số ---
    q_weights, q_biases, input_scale = get_quantized_params(model)
    conv1_w, conv2_w, dense1_w, dense2_w = q_weights
    conv1_b, conv2_b, dense1_b, dense2_b = q_biases

    # --- 3. Khởi tạo Golden Model và DUT C++ ---
    print("Initializing Golden Model and C++ DUT...")
    golden_model = GoldenModel(q_weights, q_biases)
    
    # Cần chuyển đổi định dạng NumPy sang list lồng nhau cho pybind11
    cpp_dut = cnn_dut_module.CNN_DUT(
        conv1_w.tolist(), conv1_b,
        conv2_w.tolist(), conv2_b,
        dense1_w.tolist(), dense1_b,
        dense2_w.tolist(), dense2_b
    )

    # --- 4. Chạy kiểm thử trên một ảnh ---
    print("\n--- Running Single Image Verification Test ---")
    test_idx = np.random.randint(0, x_test.shape[0])
    test_image_float = x_test[test_idx]
    true_label = y_test[test_idx]
    
    # Lượng tử hóa ảnh đầu vào và thay đổi định dạng (H, W, C) -> (C, H, W)
    test_image_q = quantize_tensor(test_image_float, input_scale).transpose(2, 0, 1)

    # Chạy Golden Model
    golden_output = golden_model.predict(test_image_q)
    golden_prediction = np.argmax(golden_output)

    # Chạy DUT C++
    dut_output = cpp_dut.predict(test_image_q.tolist())
    dut_prediction = np.argmax(dut_output)

    # --- 5. So sánh và báo cáo kết quả ---
    print(f"Test Image Index: {test_idx}")
    print(f"True Label: {true_label}")
    print("-" * 20)
    print(f"Golden Model Prediction: {golden_prediction}")
    print(f"DUT C++ Prediction:    {dut_prediction}")
    print("-" * 20)
    
    # So sánh chi tiết
    abs_diff = np.abs(np.array(dut_output) - golden_output)
    
    if dut_prediction == golden_prediction and np.all(abs_diff < 1e-3):
        print("RESULT: PASS")
        print("Predictions match and output probabilities are very close.")
    elif dut_prediction == golden_prediction:
        print("RESULT: PARTIAL PASS")
        print("Predictions match, but there are significant differences in output probabilities.")
        print(f"Max difference: {np.max(abs_diff)}")
    else:
        print("RESULT: FAIL")
        print("Predictions DO NOT match!")

    print("\nGolden Model Output:", golden_output)
    print("DUT C++ Output:   ", np.array(dut_output))


if __name__ == '__main__':
    main()