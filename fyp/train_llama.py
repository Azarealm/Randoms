import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
from datasets import Dataset as HFDataset
import numpy as np
from typing import Dict, List, Optional

# Constants
MODEL_NAME = "meta-llama/Llama-2-7b"  # Or your preferred LLaMA model
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./llama_finetuned"

def prepare_conversation_pairs(df: pd.DataFrame) -> List[Dict]:
    """
    Prepare conversation pairs with emotions from the dataframe.
    Each conversation pair consists of a user message and the next message as response.
    """
    conversation_pairs = []
    
    for i in range(0, len(df) - 1, 2):
        if i + 1 < len(df):
            conversation = {
                "user_message": df.iloc[i]["utterance"],
                "user_emotion": df.iloc[i]["emotion"],
                "assistant_message": df.iloc[i + 1]["utterance"],
                "assistant_emotion": df.iloc[i + 1]["emotion"]
            }
            conversation_pairs.append(conversation)
    
    return conversation_pairs

def format_conversation(conversation: Dict) -> str:
    """Format a single conversation pair into a training example."""
    return (
        f"User: {conversation['user_message']}\n"
        f"Emotion: {conversation['user_emotion']}\n"
        f"Assistant: {conversation['assistant_message']}\n"
        f"Emotion: {conversation['assistant_emotion']}</s>"
    )

def tokenize_function(examples: Dict, tokenizer: LlamaTokenizer) -> Dict:
    """Tokenize the texts and prepare them for training."""
    # Format each conversation
    formatted_conversations = [format_conversation(conv) for conv in examples["conversations"]]
    
    # Tokenize
    tokenized = tokenizer(
        formatted_conversations,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Set labels same as input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def prepare_dataset():
    """Prepare and process the dataset for training."""
    # Load the dataset
    df = pd.read_csv('cleaned_recon_dataset_no_punctuation.csv')
    
    # Prepare conversation pairs
    conversation_pairs = prepare_conversation_pairs(df)
    
    # Convert to HuggingFace dataset format
    dataset_dict = {"conversations": conversation_pairs}
    dataset = HFDataset.from_dict(dataset_dict)
    
    # Split into train and validation
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    return train_test_split['train'], train_test_split['test']

def train_model():
    """Initialize and train the model."""
    # Initialize tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        gradient_checkpointing=True,  # Add gradient checkpointing
        use_cache=False  # Disable KV cache during training
    )
    
    # Add special tokens
    special_tokens = {
        'pad_token': '[PAD]',
        'bos_token': '<s>',
        'eos_token': '</s>'
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    # Prepare the dataset
    train_dataset, val_dataset = prepare_dataset()
    
    # Tokenize the datasets
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="steps",
        save_total_limit=2,
        eval_steps=500,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        save_steps=1000,
        prediction_loss_only=True,
        fp16=True,  # Add mixed precision training
        gradient_accumulation_steps=4,  # Add gradient accumulation
        warmup_steps=500,  # Add warmup steps
        weight_decay=0.01,  # Add weight decay
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return model, tokenizer

# Train the model
model, tokenizer = train_model()