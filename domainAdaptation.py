from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast

# Load the pre-trained KoGPT model and tokenizer
model_name = 'skt/kogpt2-base-v2'  # Change to the appropriate GPT-2 variant
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
bos_token='</s>', eos_token='</s>', unk_token='<unk>',
pad_token='<pad>', mask_token='<mask>')
# Load and preprocess your domain-specific dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="domain_specific_data.csv",  # Path to your dataset
    block_size=128  # Adjust block_size as needed
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./domain_adaptation_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=10_000,
    dataloader_num_workers=4,
)

# Initialize Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()


model.save_pretrained("./model/my_domain_adaptation_model")
tokenizer.save_pretrained("./model/my_domain_adaptation_model")