import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
import csv

def summary2letter(summary):
    # 모델 이름 또는 모델 경로 지정
    model_name = "./model/my_fine_tuned_model"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')
    
    # 입력 텍스트
    input_text = ' '.join([text for text in summary ])
    input_text='''
    오늘 이사한 집에서 자고 일어났더니 너무 조용하고 소음에 익숙한 삶을 살았는데도 집 정리를 안 하고 왔다.'''
    # 입력 텍스트를 토큰화하고 텐서로 변환
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # 모델에 입력 전달하여 결과 생성
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # 결과 토큰을 텍스트로 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)

    letter=generated_text
    
    return letter
