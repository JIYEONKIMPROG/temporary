import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
import csv
from torch.utils.data import Dataset


class CustomDataset(Dataset):
        def __init__(self, tokenizer, inputs, max_length=128):
            self.tokenizer = tokenizer
            self.inputs = inputs
            self.max_length = max_length

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input_text = self.inputs[idx]
            output_text = self.outputs[idx]

            input_encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
           
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze()
            }



def train(kakaotalks):
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = 'skt/kogpt2-base-v2'  # Change to the appropriate GPT-2 variant
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')



    # 데이터셋 생성
    dataset = CustomDataset(tokenizer, kakaotalks)

    # DataLoader 생성
    batch_size = 32
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a DataCollator for Language Modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We are not using MLM, set to False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./my_model",
        overwrite_output_dir=True,
        num_train_epochs=30,  # Adjust the number of epochs
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )


    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./model/my_fine_tuned_model")
    tokenizer.save_pretrained("./model/my_fine_tuned_model")
    
    
'''

model_name = 'skt/kogpt2-base-v2'  # Change to the appropriate GPT-2 variant
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

    csv_file_path = '../data/summaryLetter.csv'

    summary_column = 'summary'
    letter_column='letter'
    input_sentences = []
    output_letters=[]

    # CSV 파일 열기
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # CSV 파일을 반복하면서 특정 열의 데이터 추출
        for row in reader:
            summary = row.get(summary_column)
            letter = row.get(letter_column)
            if summary is not None:
                input_sentences.append(summary)
            if letter is not None:
                output_letters.append(letter)

    from torch.utils.data import Dataset, DataLoader
    class CustomDataset(Dataset):
        def __init__(self, tokenizer, inputs, outputs, max_length=128):
            self.tokenizer = tokenizer
            self.inputs = inputs
            self.outputs = outputs
            self.max_length = max_length

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input_text = self.inputs[idx]
            output_text = self.outputs[idx]

            input_encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            output_encoding = self.tokenizer(output_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': output_encoding['input_ids'].squeeze(),
            }

    # 데이터셋 생성
    dataset = CustomDataset(tokenizer, input_sentences,output_letters)

    # DataLoader 생성
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a DataCollator for Language Modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We are not using MLM, set to False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./my_model",
        overwrite_output_dir=True,
        num_train_epochs=30,  # Adjust the number of epochs
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )


    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./my_fine_tuned_model")
    tokenizer.save_pretrained("./my_fine_tuned_model")

'''