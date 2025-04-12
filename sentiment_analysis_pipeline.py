from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd

# Step 1: Loading Excel file
df = pd.read_excel('synthetic_reviews_dataset.xlsx')

# Step 2: Converting ratings to labels
# 1 (positive), 0 (negative), 2 (neutral)
df['label'] = df['Rating'].apply(lambda r: 1 if r >= 4 else 0 if r <= 2 else 2)

# Step 3: Rename 'Comment' to 'review'
df = df.rename(columns={'Comment': 'review'})

# Step 4: Drop any rows with missing reviews or labels
df = df.dropna(subset=['review', 'label'])

# Step 5: Converting to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['review', 'label']])

# Step 6: Loading BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Step 7: Tokenization function
def tokenize_function(batch):
    return tokenizer(batch['review'], truncation=True)

# Step 8: Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 9: Spliting into train and test sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Step 10: Format the dataset for PyTorch
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 11: Use dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 12: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_total_limit=1,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",  
)

# Step 13: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 14: Train the model
trainer.train()

# Step 15: Save the model and tokenizer
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
