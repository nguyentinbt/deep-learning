# Khai báo hàm
def calculate_accuracy(correct_prediction, total_samples):
    """
    Đầu vào (Input):
    - correct_prediction: Số lượng mô hình đoán đúng
    - total_samples: tổng số dữ liệu kiểm tra
    """

    # Xử lý logic
    accuricy = correct_prediction / total_samples

    return  accuricy

# sử dụng hàm
model_accuricy = calculate_accuracy(95, 100) 
print("Độ chính xác mô hình: ", model_accuricy)


def calculate_error(y_true,y_pred):
    MAE = abs(y_true - y_pred)
    return  MAE

MAE_result = calculate_error(10, 7)
print("Sai số tuyệt đối:", MAE_result)

