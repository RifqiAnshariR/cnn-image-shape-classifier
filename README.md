# 📌 Description  
**Image Shape Classifier**  
This project implements an image shape classifier using a **CNN Algorithm** to classify shapes: **Triangle, Circle, and Rectangle**. The shapes are generated randomly with a **Grayscale color scheme**.  

# 🚀 How to Run  

### **1. Set up the virtual environment**  
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### **2. Install dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Prepare the dataset**  
```bash
python prep.py
```

### **4. Train the model**  
```bash
python train.py
```

### **5. Test the model**  
```bash
python test.py
```

# ➕ Miscellaneous  
- Classification report:

| Class       | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Circle**    | 0.95      | 0.97   | 0.96     | 37      |
| **Rectangle** | 0.98      | 0.96   | 0.97     | 45      |
| **Triangle**  | 1.00      | 1.00   | 1.00     | 38      |

- Overall Metrics:

**Accuracy**: 0.97 (120 samples), **Macro Avg**: Precision 0.97, Recall 0.98, F1-Score 0.98, **Weighted Avg**: Precision 0.98, Recall 0.97, F1-Score 0.98

---  
Made with ❤️!