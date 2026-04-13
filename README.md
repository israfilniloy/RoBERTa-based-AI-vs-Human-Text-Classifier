# RoBERTa AI vs Human Text Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Model-RoBERTa-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Text%20Classification-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-HuggingFace%20Transformers-orange?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Enabled-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>

---

## 📌 Overview

With the rapid rise of AI-generated content, distinguishing between **human-written** and **AI-generated** text has become a critical challenge in academia, journalism, and content moderation.

This project fine-tunes **RoBERTa-base** (`roberta-base`) — a robustly optimized BERT pretraining approach — on the [AI and Human Generated Text dataset](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text) from Hugging Face to perform **binary text classification**:

- **Label 0** → Human-written text
- **Label 1** → AI-generated text

The model is trained for **10 epochs** using the HuggingFace `Trainer` API with F1-score as the primary optimization metric.

---

## 🏗️ Model Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────┐
│     Text Preprocessing      │
│  • Lowercase                │
│  • Remove URLs              │
│  • Remove special chars     │
│  • Remove extra whitespace  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     RoBERTa Tokenizer       │
│  • Max length: 512 tokens   │
│  • Padding & Truncation     │
│  • Vocab size: 50,265       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   RoBERTa-base Encoder      │
│  • 125M parameters          │
│  • 12 transformer layers    │
│  • 768 hidden dimensions    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Classification Head        │
│  • 2 output labels          │
│  • Sigmoid activation       │
└──────────────┬──────────────┘
               │
               ▼
    Human (0) / AI (1)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Base Model | `roberta-base` |
| Epochs | 10 |
| Train Batch Size | 8 |
| Eval Batch Size | 16 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Max Token Length | 512 |
| Best Model Metric | F1-Score |
| Evaluation Strategy | Per epoch |
| Train/Test Split | 80% / 20% |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Ateeqq/AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text) |
| Platform | Hugging Face Datasets |
| Labels | 0 = Human, 1 = AI-generated |
| Split | 80% train / 20% test (seed=42) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/israfilniloy/RoBERTa-AI-Human-Classifier.git
cd RoBERTa-AI-Human-Classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook roberta.ipynb
```

**Run cells in order:**
1. **Install packages** — installs transformers, datasets, accelerate
2. **Import libraries** — loads all dependencies
3. **Load dataset** — pulls from Hugging Face Hub automatically
4. **Clean text** — lowercase, remove URLs and special characters
5. **Split dataset** — 80/20 train/test split
6. **Load tokenizer** — RoBERTa tokenizer with max length 512
7. **Tokenize** — encodes text for model input
8. **Load model** — RoBERTa-base with classification head
9. **Define metrics** — accuracy, precision, recall, F1
10. **Train** — 10 epochs with HuggingFace Trainer
11. **Evaluate** — final metrics on test set
12. **Confusion matrix** — visual evaluation
13. **Save model** — saves to `./roberta_final_model/`

> 💡 **GPU recommended** — The notebook auto-detects CUDA. Training on CPU will be very slow for 10 epochs.

---

## 🔮 Run Inference on New Text

After training, load the saved model and classify new text:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load saved model
model = AutoModelForSequenceClassification.from_pretrained('./roberta_final_model')
tokenizer = AutoTokenizer.from_pretrained('./roberta_final_model')

model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    label = "🤖 AI-Generated" if pred == 1 else "👤 Human-Written"
    return label

# Example
text = "The rapid advancement of large language models has transformed..."
print(predict(text))
```

---

## ⚙️ Requirements

| Package | Purpose |
|---|---|
| `transformers` | RoBERTa model & tokenizer |
| `datasets` | HuggingFace dataset loading |
| `torch` | PyTorch backend |
| `accelerate` | Training acceleration |
| `scikit-learn` | Evaluation metrics |
| `pandas` | Data handling |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix visualization |

---

## 📬 Citation

If you use this work, please cite:

```bibtex
@misc{niloy2026roberta,
  title   = {RoBERTa-based AI vs Human Text Classifier},
  author  = {Niloy, Al Israfil},
  year    = {2026},
  url     = {https://github.com/israfilniloy/RoBERTa-AI-Human-Classifier}
}
```

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) for the RoBERTa model and Trainer API
- [Ateeqq](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text) for the AI & Human Generated Text dataset
- American International University-Bangladesh (AIUB) for research support

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
