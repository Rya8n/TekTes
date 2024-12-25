import joblib
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('wklyStudyHours_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

mathScore = input("Please input math score: ")
readingScore = input("Please input reading score: ")
writingScore = input("Please input writing score: ")

if isinstance(mathScore, int) and isinstance(readingScore, int) and isinstance(writingScore, int):
    new_data = np.array([[mathScore, readingScore, writingScore]])  
    y_pred = model.predict(new_data)
    predicted_class_index = np.argmax(y_pred, axis=1)[0]
    predicted_grade = label_encoder.inverse_transform([predicted_class_index])
    print(f"Predicted Grade: {predicted_grade[0]}")
else:
    "Data invalid. Please re-run the program"