import os
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path = ""
save_path = ""
file_list = os.listdir(file_path)
os.makedirs(save_path, exist_ok=True)  # 关键修复：添加 exist_ok=True
print(file_list)

for file_name in file_list:
    input_file = os.path.join(file_path, file_name)
    output_file = os.path.join(save_path, file_name)
    print(input_file)

    try:
        original_data = np.genfromtxt(input_file, delimiter=",")
        X = original_data[:, :-1]
        y = original_data[:, -1]

        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(X)

        new_y = y.reshape(-1, 1)
        processed_data = np.hstack((X_scaler, new_y))

        np.savetxt(output_file, processed_data, fmt='%.6f', delimiter=',')

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        continue