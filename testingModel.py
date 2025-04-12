from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./bert_sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label map
label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}

# Function to map rating to expected sentiment
def rating_to_sentiment_label(rating):
    if rating >= 4:
        return 1  # Positive
    elif rating <= 2:
        return 0  # Negative
    else:
        return 2  # Neutral

# Predict sentiment from review
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class, label_map[predicted_class]

#  GIVE YOUR INPUTS HERE 
inputs = [
    # {"review": "Very bad experience, not worth the price", "rating": 5},
    # {"review": "Amazing product! Loved it", "rating": 1},
    # {"review": "Just okay", "rating": 3},
    # {"review": "Worst service ever", "rating": 1},
    # {"review": "Excellent quality and fast delivery", "rating": 5},
    {"review": "Very good experience", "rating": 1},
    {"review": "Amazing product! Loved it", "rating": 1},
    {"review": "Just okay", "rating": 1},
    {"review": "Worst service ever", "rating": 1},
    {"review": "Excellent quality and fast delivery", "rating": 2},
    {"review": "Not good but packaging is good", "rating": 2},
]

# Process each input
for i, item in enumerate(inputs):
    review = item["review"]
    rating = item["rating"]
    expected_label = rating_to_sentiment_label(rating)

    predicted_label, predicted_name = predict_sentiment(review)
    expected_name = label_map[expected_label]
    conflict = predicted_label != expected_label

    print(f"\n Review #{i+1}")
    print(f"     Review: {review}")
    print(f"     Rating: {rating}")
    print(f"    Predicted Sentiment: {predicted_name}")
    print(f"    Expected from Rating: {expected_name}")
    print(f"    Conflict: {'YES ' if conflict else 'No'}")
