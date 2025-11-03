# 😊 MoodVision — AI Emotion Detection App

A real-time emotion recognition tool built with **Streamlit** that identifies human emotions from **uploaded images** or **live webcam feeds** using a Vision Transformer (ViT) model.

🔗 **Live Demo:** [MoodVision on Streamlit](https://image-emotion-detection-app.streamlit.app/)

---

## ✨ Features

- 🖼️ **Upload Images** — Detect emotions instantly from photos.
- 🎥 **Live Camera Mode** — Analyze emotions in real time through your webcam.
- 🧠 **AI-Powered Predictions** — Uses a fine-tuned ViT model for high accuracy.
- 🗂️ **Detection History** — Automatically logs all detections for later viewing.
- 💾 **Local Database** — Stores results securely in a lightweight SQLite database.

---

## 😃 Emotions Recognized

The model can classify **seven** distinct emotions:

| Emotion  | Emoji |
| -------- | :---: |
| Angry    |  😠   |
| Disgust  |  🤢   |
| Fear     |  😨   |
| Happy    |  😊   |
| Sad      |  😢   |
| Surprise |  😲   |
| Neutral  |  😐   |

---

## 🧰 Tech Stack

- **Python 3.x**
- **Streamlit** — Web app framework
- **PyTorch** — Deep learning backend
- **Transformers (Hugging Face)** — Pretrained ViT model
- **OpenCV** — Image and camera processing
- **SQLite** — Lightweight local database
- **Pillow (PIL)** — Image manipulation library

---

## ⚙️ Installation Guide

1. Clone the repository:

```bash
git clone https://github.com/vic1500/image-emotion-project.git
cd image-emotion-project
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Author

**Fele Adedoyin**  
📚 _Industrial Mathematics (Computer Science option) Student_

💼 **LinkedIn:** [https://www.linkedin.com/in/adedoyin-fele-117286247/]  
🐙 **GitHub:** [https://github.com/Adedoyin-fl]  
✉️ **Email:** [adedoyinfele04@gmail.com]
