import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sys
import os

# --- set up---
# Adding path to C++ module (build folder)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
# Adding current path to import golden_model.py
sys.path.append(os.path.dirname(__file__))

try:
    import cnn_dut_module
    from golden_model import GoldenModel
except ImportError:
    st.error(
    "ERROR: Failed to import the C++ module (cnn_dut_module).\n"
    "Make sure you have successfully compiled the C++ code using CMake and Make.\n"
    "Run the following commands from the project root directory:\n"
    "1. mkdir build\n"
    "2. cd build\n"
    "3. cmake ..\n"
    "4. make"
    )
    st.stop()

# --- quantize_tensor ---

def quantize_tensor(tensor, scale_factor):
    """Lượng tử hóa một tensor float32 thành int8."""
    quantized = np.round(tensor * scale_factor)
    return np.clip(quantized, -128, 127).astype(np.int8)

def quantize_bias(tensor, input_scale, weight_scale):
    """Lượng tử hóa bias."""
    return (tensor * input_scale * weight_scale).astype(np.int32)

# --- Using cache of  Streamlit ---
@st.cache_resource
def load_all_models_and_data():
    """
        This function will load the data, train or load the Keras model, 
        quantize the weights, and initialize both the Golden Model and the C++ DUT. The results will be cached to improve speed."
    """
    with st.spinner('Data Loading and creating models.'):
        # 1. Data loading
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        img_rows, img_cols = 28, 28
        x_test_float = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0

        # 2. Download or train the model
        model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.h5")
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
        else:
            # runtest_train.py to trainning the model
            st.error(f"we couldn't find the model file '{model_path}'. Please run`python python_testbench/run_test.py` first.")
            st.stop()
        
        # 3. extraction and quantization
        input_scale = 127.0
        # Adding np.abs() 
        conv1_w_scale = 127.0 / np.max(np.abs(model.layers[1].get_weights()[0]))
        conv2_w_scale = 127.0 / np.max(np.abs(model.layers[2].get_weights()[0]))
        dense1_w_scale = 127.0 / np.max(np.abs(model.layers[6].get_weights()[0]))
        dense2_w_scale = 127.0 / np.max(np.abs(model.layers[7].get_weights()[0]))
        
        conv1_w_f, conv1_b_f = model.layers[1].get_weights()
        conv1_w_q = quantize_tensor(conv1_w_f, conv1_w_scale).transpose(3, 2, 0, 1)
        conv1_b_q = quantize_bias(conv1_b_f, input_scale, conv1_w_scale)

        conv2_w_f, conv2_b_f = model.layers[2].get_weights()
        conv2_w_q = quantize_tensor(conv2_w_f, conv2_w_scale).transpose(3, 2, 0, 1)
        conv2_b_q = quantize_bias(conv2_b_f, 1.0, conv2_w_scale)

        dense1_w_f, dense1_b_f = model.layers[6].get_weights()
        dense1_w_q = quantize_tensor(dense1_w_f, dense1_w_scale)
        dense1_b_q = quantize_bias(dense1_b_f, 1.0, dense1_w_scale)

        dense2_w_f, dense2_b_f = model.layers[7].get_weights()
        dense2_w_q = quantize_tensor(dense2_w_f, dense2_w_scale)
        dense2_b_q = quantize_bias(dense2_b_f, 1.0, dense2_w_scale)
        
        q_weights = [conv1_w_q, conv2_w_q, dense1_w_q, dense2_w_q]
        q_biases = [conv1_b_q.tolist(), conv2_b_q.tolist(), dense1_b_q.tolist(), dense2_b_q.tolist()]

        # 4. Initilize Golden Model and C++ DUT
        golden_model = GoldenModel(q_weights, q_biases)
        cpp_dut = cnn_dut_module.CNN_DUT(
            conv1_w_q.tolist(), q_biases[0],
            conv2_w_q.tolist(), q_biases[1],
            dense1_w_q.tolist(), q_biases[2],
            dense2_w_q.tolist(), q_biases[3]
        )
        
        return x_test_float, y_test, golden_model, cpp_dut, input_scale

# --- New function to test all function ---
@st.cache_data
def run_regression_test(_golden_model, _cpp_dut, _x_test, _y_test, _input_scale, num_samples):
    """
    Run tests on a subset of the test dataset and return statistics.
    Use cache_data to avoid rerunning when not necessary.
    """
    pass_count = 0
    partial_pass_count = 0
    fail_count = 0
    
    # Limit amount in the website
    indices = np.random.choice(len(_x_test), num_samples, replace=False)
    
    progress_bar = st.progress(0, text="Đang chạy kiểm tra toàn diện...")
    
    for i, idx in enumerate(indices):
        test_image_float = _x_test[idx]
        
        # quantilization
        test_image_q = quantize_tensor(test_image_float, _input_scale).transpose(2, 0, 1)

        # Run both models
        golden_output = _golden_model.predict(test_image_q)
        golden_prediction = np.argmax(golden_output)
        
        dut_output = _cpp_dut.predict(test_image_q.tolist())
        dut_prediction = np.argmax(dut_output)
        
        # classification result comparison
        abs_diff = np.abs(np.array(dut_output) - golden_output)
        if dut_prediction == golden_prediction:
            if np.all(abs_diff < 1e-3):
                pass_count += 1
            else:
                partial_pass_count += 1
        else:
            fail_count += 1
            
        #  Up date the progress bar
        progress_bar.progress((i + 1) / num_samples, text=f"Đã kiểm tra {i+1}/{num_samples} ảnh...")

    progress_bar.empty() # Xóa thanh tiến trình khi hoàn tất
    return pass_count, partial_pass_count, fail_count


# --- Streamlit interface ---
# --- Main streamlit interface---
st.set_page_config(layout="wide")
st.title("👨‍🏫  Comprehensive Testbench for Number Recognition AI Chip (CNN)")

# Download data, quantize weights, and initialize both the Golden Model and the C++ DUT
x_test, y_test, golden_model, cpp_dut, input_scale = load_all_models_and_data()

# --- seperate  ---
tab1, tab2 = st.tabs(["🔬 Images check (Debugger)", "📈 Full check (Regression)"])

# === TAB 1: Check an Image ===
with tab1:
    st.header("Check and visualize results on a single image")
    
    # Thanh điều khiển bên trái
    st.sidebar.header(" Debugger controller")
    test_idx = st.sidebar.slider(
        "Choose an image to test:",
        0, len(x_test) - 1, 100
    )

    # take the image
    test_image_float = x_test[test_idx]
    true_label = y_test[test_idx]

    # Run for a test
    test_image_q = quantize_tensor(test_image_float, input_scale).transpose(2, 0, 1)
    golden_output = golden_model.predict(test_image_q)
    golden_prediction = np.argmax(golden_output)
    dut_output = cpp_dut.predict(test_image_q.tolist())
    dut_prediction = np.argmax(dut_output)

    # Show picture and results
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(test_image_float, caption=f"input image #{test_idx}", width=250)
        st.info(f"**True Lable: {true_label}**")

    with col2:
        st.subheader("compare results")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Golden Model predict", f"{golden_prediction}")
        metric_col2.metric("Chip C++", f"{dut_prediction}")

        abs_diff = np.abs(np.array(dut_output) - golden_output)
        max_diff = np.max(abs_diff)
    # Analysis the dashboard
    st.subheader("out put probability vector")
    # --- Specifict Analysis---
    st.subheader("Analysis")
    st.markdown("Table under shows the output probabilities from both models and the absolute differences.")

    df = pd.DataFrame({
        'The number': list(range(10)),
        'Golden Model (Python)': golden_output,
        'DUT (C++)': dut_output,
        'Absolute error': abs_diff
    })

    st.dataframe(df.style.format({
        'Golden Model (Python)': '{:.4f}',
        'DUT (C++)': '{:.4f}',
        'Absolute error': '{:.4f}'
    }).background_gradient(cmap='Reds', subset=['Absolute error']))

    st.bar_chart(df.set_index('The number')[['Golden Model (Python)', 'DUT (C++)']])


# TAB 2: Check Full round 
with tab2:
    st.header("Run on multiple images and get statistics")
    st.sidebar.header("Regression controller")
    num_samples_to_test = st.sidebar.number_input(
        "Number of random images to test:", 
        min_value=10, max_value=1000, value=100, step=10
    )

    if st.sidebar.button("🚀 start "):
        pass_count, partial_pass_count, fail_count = run_regression_test(
            golden_model, cpp_dut, x_test, y_test, input_scale, num_samples_to_test
        )
        
        st.subheader(f"resuls on  {num_samples_to_test} random images")
        st.markdown(f"- ✅ PASS: {pass_count} images")
        st.markdown(f"- ⚠️ PARTIAL PASS: {partial_pass_count} images")
        st.markdown(f"- ❌ FAIL: {fail_count} images")
        
        # Show pie chart
        df_results = pd.DataFrame({
            'Type of results': ['PASS', 'PARTIAL PASS', 'FAIL'],
            'amount': [pass_count, partial_pass_count, fail_count]
        })
        
        # using mathplotlib (pie chart)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.pie(df_results['amount'], labels=df_results['Type of results'], autopct='%1.1f%%',
               colors=['#4CAF50', '#FFC107', '#F44336'], startangle=90)
        ax.axis('equal')  
        st.pyplot(fig)

        st.metric("Percent successful (PASS + PARTIAL PASS)", 
                  f"{((pass_count + partial_pass_count) / num_samples_to_test) * 100:.2f}%")
        
        st.markdown("---")
        st.subheader("Explain the difference")
        st.markdown( """" The difference between the 'Teacher' (Golden Model) and the 'Student' (C++ Chip) results mainly comes from errors that were intentionally injected into the C++ code to simulate real chip-design issues.

 ### 1. Overflow Error (Overflow) - Primary cause of FAIL

Issue: In Conv2D and Fully Connected layers, multiplication and accumulation operations occur. To save resources, the 'chip' uses a limited data type (int16_t) for accumulators. When inputs or weights are large, the accumulator can exceed its limit (e.g., > 32767), causing an overflow and producing completely incorrect results.

Example: Imagine a scale that can only measure up to 10 kg. If you place 12 kg on it, it may show a wildly wrong number (for example 2 kg or a negative number) instead of signaling an error.

Impact: This is the most severe bug and often leads to incorrect predictions (FAIL).

### 2. Quantization Logic Bug - Primary cause of PARTIAL PASS

Issue: After each compute layer, results must be rounded or truncated back to 8-bit (int8_t) for the next layer. In the C++ code this truncation logic is implemented incorrectly. For example, instead of allowing a maximum of 127, it is limited to 125.

Impact: This bug does not cause immediate failures but introduces a small error. That error accumulates over many layers and skews the final probability vector of the C++ chip compared to the Golden Model, even if the final predicted class may remain correct. This is the main reason for PARTIAL PASS.

### 3. Pooling Size Calculation Bug

Issue: The logic that computes the output size of MaxPooling in C++ may be slightly wrong (for example, incorrect rounding when the size is not evenly divisible).

Impact: This changes the shape of the feature map, causing the subsequent Flatten layer to produce a vector of different length or scrambled values, which breaks the rest of the computation.""")