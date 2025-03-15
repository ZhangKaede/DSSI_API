import numpy as np
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io

# 生成示例数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建深度学习模型
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.get_json(force=True)
        input_values = data.get('input')
        if input_values is None:
            return jsonify({"error": "Missing 'input' in request data"}), 400
        input_array = np.array(input_values).reshape(-1, 1)
        predictions = model.predict(input_array)

        # 绘制预测图
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, color='blue', label='Training data')
        plt.scatter(input_array, predictions, color='red', label='Predictions')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Deep Learning Predictions')
        plt.legend()

        # 将图像保存到内存中的缓冲区
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close()

        # 返回预测结果和图像
        result = {
            "predictions": predictions.flatten().tolist(),
            "image": send_file(img_buf, mimetype='image/png')
        }
        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    