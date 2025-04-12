
# Sentiment Analysis on Product Reviews using BERT

This project leverages **BERT (Bidirectional Encoder Representations from Transformers)** to detect sentiment mismatches in product reviews based on their associated star ratings. It helps uncover **genuine vs non-genuine reviews** by comparing model-predicted sentiment with user-provided star ratings.

---

## 🧠 Project Objective

To build a fine-tuned BERT model that classifies sentiment (Positive, Negative, Neutral) of product reviews and flags **conflicts** where:

> The textual sentiment does **not align** with the numerical rating  
> (e.g., rating = ⭐1 but review = "Amazing product!")

---

## 📂 Project Structure

```
BERT_MODEL/
├── bert_sentiment_model/          # Final trained model + tokenizer
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── logs/                          # Training logs for TensorBoard
│   └── events.out.tfevents.*
│
├── results/checkpoint-1800/       # Intermediate checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   ├── training_args.bin
│   └── vocab.txt
│
├── synthetic_reviews_dataset.xlsx # Dataset (reviews + ratings)
├── sentiment_analysis_pipeline.py # Fine-tuning script for BERT
└── testingModel.py                # Script to test and detect mismatches
```

---

## 📊 Dataset Format

The dataset is stored in `synthetic_reviews_dataset.xlsx` with the following columns:

| Customer Name | Review Title | Rating | Comment |
|---------------|---------------|--------|---------|

During preprocessing:
- **Comment** → renamed to `review`
- **Rating** → converted to `label`:  
  - `1` = Positive (rating ≥ 4)  
  - `0` = Negative (rating ≤ 2)  
  - `2` = Neutral (rating = 3)

---

## 🔧 How to Train the Model

```bash
python sentiment_analysis_pipeline.py
```

This script:
- Loads the Excel dataset
- Preprocesses & tokenizes reviews using `bert-base-uncased`
- Fine-tunes a BERT model using Hugging Face's `Trainer`
- Saves model and tokenizer to `bert_sentiment_model/`

---

## 🧪 How to Test & Detect Review Mismatches

```bash
python testingModel.py
```

This script:
- Loads the trained model and tokenizer
- Takes hardcoded reviews + ratings
- Predicts sentiment from review
- Compares with expected label derived from rating
- Logs any **conflicts** (e.g., positive review + low rating)

### Sample Output:
```
Review #1
     Review: Very good experience
     Rating: 1
     Predicted Sentiment: Positive
     Expected from Rating: Negative
     Conflict: YES
```

---

## 📦 Dependencies

- `transformers`
- `datasets`
- `torch`
- `pandas`
- `openpyxl` (for reading Excel files)

Install all with:

```bash
pip install -r requirements.txt
```

> Create a `requirements.txt` if needed:

```txt
transformers
datasets
torch
pandas
openpyxl
```

---

## 📈 Applications

- Detect **fake or suspicious reviews**
- Improve **rating credibility**
- Automate **review moderation**
- Power intelligent **sentiment dashboards**

---

## 🙌 Contribution

Feel free to fork, improve, or extend the model with:
- Larger datasets
- Real-world review sources (Flipkart, Amazon, etc.)
- More granular sentiment (e.g., 5-class)

---

## 📩 Contact

Made with ❤️ by Utam Kumar(m23aid063@iitj.ac.in) 
- Open for collaborations or contributions!
