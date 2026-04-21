# Acne Detection & Skin Analyzer

## Overview

This project is a simple image-processing application that analyzes a facial photo and estimates acne severity.

It works by:

* detecting the face
* isolating the facial region
* analyzing redness and texture
* computing a severity score
* classifying the result as **Mild, Moderate, or Severe**

---

## Features

* Upload a face image (JPG/PNG)
* Automatic face detection and cropping
* Red/inflamed region detection
* Basic texture analysis
* Severity scoring
* Visual output
---

## Project Structure

```
acne-detector/
│
├── app/
│   ├── acne_features.py
│   ├── face_detect.py
│   ├── main.py
│   ├── preprocess.py
│   ├── severity.py
│   └── utils.py
│
├── notebooks/
│   ├── Acne Detection & Skin Analyzer.ipynb
│  
│
├── data/
│   ├── samples/
│   └── test/
│
├── outputs/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/acne-detector.git
cd acne-detector
```

---

### 2. Create virtual environment

#### Windows (CMD):

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the App

### Option 1 — Streamlit 

```bash
streamlit run app/main.py
```

Then open the browser and:

* upload an image
* click analyze
* view results

---

### Option 2 — Run as script

```bash
python app/main.py
```

(This runs a basic pipeline without UI)

---

## Input Requirements

Use images that are:

* clear
* well-lit
* visible face
* mostly frontal or slightly angled

---

## Limitations

* Not medically accurate
* Sensitive to lighting conditions
* This Project was not made with a trained model

---

## Future Improvements

* Better skin masking
* Mobile or webcam support

---

## Tech Used

* Python
* OpenCV
* NumPy
* Matplotlib
* Streamlit
* Jupyter Notebook
