# 🔥 Violence Detection Model (violencemodel.py)

This module implements a deep learning model for video-based violence detection, using a VGG19 + LSTM architecture to capture both spatial and temporal features.

It includes:

A preprocessing utility to read and normalize video data.

A model builder that constructs the architecture and loads pretrained weights.

A prediction helper that detects violence probability in short video clips.

## 🧩 Features

✅ VGG19-based CNN extracts frame-level spatial features.

🔁 LSTM layer models temporal dependencies across frames.

📈 Binary classification — predicts Violence vs No Violence.

💾 Automatically loads pretrained weights from fightw.h5.

🧠 Ready for inference or further fine-tuning.

🧠 Model Architecture

## The model combines:

CNN Backbone: VGG19 (pretrained on ImageNet, without top layers)

LSTM Layer: Captures motion and temporal context

Dense Layers: Fully connected classifier with dropout regularization

Input (30 frames of 160×160×3)
   ↓
TimeDistributed(VGG19)
   ↓
LSTM(30 units)
   ↓
Dense(90) → Dropout(0.1)
   ↓
GlobalAveragePooling1D
   ↓
Dense(512, relu) → Dropout(0.3)
   ↓
Dense(2, sigmoid)

## 📦 Dependencies

Install the required libraries before use:

pip install tensorflow opencv-python scikit-image numpy

## ⚙️ Functions Overview
🧾 video_reader(filename, sequence_length=30, target_size=(160,160))

Reads and preprocesses a video file for prediction.

Returns:
A NumPy array of shape (1, sequence_length, 160, 160, 3).

Example:

from violencemodel import video_reader

clip = video_reader("sample_video.mp4", sequence_length=30)

🧠 souhaiel_model(weights_path='fightw.h5')

Builds and compiles the VGG19 + LSTM model.
If the specified weights file exists, it loads pretrained parameters automatically.

Returns:
A compiled tf.keras.Model ready for prediction.

Example:

from violencemodel import souhaiel_model

model = souhaiel_model("fightw.h5")

🔍 pred_fight(model, video_array, threshold=0.9)

Performs violence prediction on a preprocessed video array.

Parameters:

model: Loaded Keras model

video_array: Array of shape (1, seq_len, H, W, 3)

threshold: Probability threshold for “Violence” classification (default: 0.9)

Returns:
(is_violence: bool, probability: float)

Example:

from violencemodel import video_reader, souhaiel_model, pred_fight

clip = video_reader("fight_scene.mp4")
model = souhaiel_model("fightw.h5")

violence_detected, prob = pred_fight(model, clip)
print(f"Violence: {violence_detected} (p={prob:.2f})")

## 🧪 Example Output
[INFO] Loaded weights from /path/to/fightw.h5
Violence: True (p=0.93)

## 🧠 Training Note

While this module focuses on inference, the architecture is suitable for training using standard Keras workflows.
You can retrain it on your own labeled dataset of violent/non-violent clips by calling:

model = souhaiel_model()
model.fit(X_train, y_train, batch_size=4, epochs=20)
model.save_weights("fightw_new.h5")

## ⚠️ Notes

The model expects exactly 30 frames per clip. For longer videos, sample or segment them.

Ensure the fightw.h5 weights file is compatible with this architecture.

The model outputs probabilities in [0, 1]; by default, values ≥ 0.9 indicate Violence.

The frame preprocessing uses anti-aliasing resizing from scikit-image.

## 📜 License

Distributed under the MIT License — free for research and educational use.

## 👨‍💻 Author

Julian Garrett/Aliniant Labs
Computer Vision & Deep Learning Researcher
📧 Julian.garrett@aliniant.com
💼 [GitHub / LinkedIn / Portfolio link]

# Massive props to a dude named Souhaeil without whom this wouldn't have been done

