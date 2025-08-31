# 👗 ChromaShape – Your Personal AI Stylist  

ChromaShape is a **Flask-based AI-powered fashion assistant** that analyzes your **body shape, skin tone, undertone, hair color, and eye color** from uploaded images. It then cross-references these features with a curated fashion dataset to provide **personalized style suggestions** including clothing colors, patterns, accessories, and outfit recommendations.  

---

## ✨ Features  

- 📸 **Upload Images**: Upload a head-to-knee body photo and a wrist photo for analysis.  
- 🤖 **AI Vision Analysis**: Uses **OpenCV + Mediapipe** for body and facial landmark detection.  
- 🎨 **Color & Feature Extraction**: Detects hair color, eye color, skin tone, undertone.  
- 📐 **Body Shape Detection**: Supports both **manual landmark marking** and **auto-detection**.  
- 👕 **Personalized Suggestions**: Recommends clothing styles, colors, fabrics, and accessories.  
- 📄 **Export Reports**: Download results as **PDF or Image** from the results page.  
- 🔒 **Privacy Friendly**: Uploaded images are processed temporarily and not stored permanently.  

---

## 🛠️ Tech Stack  

- **Frontend**: HTML5, CSS3, JavaScript  
- **Backend**: Flask (Python)  
- **Computer Vision**: OpenCV, Mediapipe  
- **ML/Clustering**: scikit-learn (MiniBatchKMeans)  
- **Data Handling**: Pandas, NumPy   

### 1. Clone the Repository  
```bash
git clone https://github.com/yourusername/chromashape.git
cd chromashape

