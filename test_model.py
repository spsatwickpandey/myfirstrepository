from model.model import load_model, load_scaler
import numpy as np

def test_prediction():
    model = load_model()
    scaler = load_scaler()
    test_input = np.array([1, 30, 0, 0, 50, 1, 1, 0]).reshape(1, -1)
    prediction = model.predict(scaler.transform(test_input))
    print("Test prediction successful!")
    print(f"Prediction result: {'Survived' if prediction[0] == 1 else 'Not Survived'}")

if __name__ == "__main__":
    test_prediction() 