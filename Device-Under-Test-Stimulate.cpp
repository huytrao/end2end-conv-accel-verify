// Sitimulart
#include <iostream>
#include <vector>
#include <cstdint>  // dùng int8_t, int16_t
#include <cstdlib>  // rand()

using namespace std;

// Stimulate a hardware accelerator for low-precision matrix multiplication
class MatrixAccelerator {
public:
    // Function multiply two matrices with simulated hardware faults
    vector<vector<int16_t>> multiply(const vector<vector<int8_t>>& A,
                                     const vector<vector<int8_t>>& B) {
        int rowsA = A.size();
        int colsA = A[0].size();
        int colsB = B[0].size();

        vector<vector<int16_t>> result(rowsA, vector<int16_t>(colsB, 0));

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                int16_t sum = 0;
                for (int k = 0; k < colsA; ++k) {
                    int16_t product = A[i][k] * B[k][j];

                    // stimulate (quantization)
                    if (product > 127) product = 127;
                    if (product < -128) product = -128;

                    sum += product;

                    // ⚠️ Lỗi phần cứng mô phỏng: overflow không được kiểm tra
                    // Nếu sum > 255 hoặc < -255 thì sẽ tự động tràn trong int16_t
                }

                // ⚠️ Giả lập lỗi logic phần cứng: đôi khi bỏ qua 1 phần tử
                if (rand() % 20 == 0) { 
                    sum -= 5; // gây sai lệch nhỏ
                }

                result[i][j] = sum;
            }
        }

        return result;
    }
};

// print matrix function
template<typename T>
void printMatrix(const vector<vector<T>>& M, const string& name) {
    cout << name << ":\n";
    for (const auto& row : M) {
        for (auto val : row)
            cout << (int)val << "\t";
        cout << "\n";
    }
    cout << endl;
}

// ========================== MAIN ==========================
int main() {
    // Create input matrix to test
    vector<vector<int8_t>> A = {
        {10, 20, 30},
        {40, 50, 60}
    };

    vector<vector<int8_t>> B = {
        {1, 2},
        {3, 4},
        {5, 6}
    };

    MatrixAccelerator dut; 
    auto result = dut.multiply(A, B);

    printMatrix(A, "Matrix A");
    printMatrix(B, "Matrix B");
    printMatrix(result, "Result (with simulated hardware faults)");

    return 0;
}
