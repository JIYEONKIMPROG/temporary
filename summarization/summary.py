from transformers import AutoTokenizer, BartForConditionalGeneration
import re

def get_summarization(dialogue_day):
    
    model_name = "alaggung/bart-r3f"
    max_length = 64
    num_beams = 5
    length_penalty = 1.2

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    
    try:
        inputs = tokenizer("[BOS]" + "[SEP]".join(dialogue_day) + "[EOS]", return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
            use_cache=True,
        )
        summarization = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summarization
    except:
        pass

def summary(dialogue):
        
    summary_list=[]
    dialogue_day=[]
    
    pattern_to_remove_date = "-{15}\s\d+년\s\d+월\s\d+일\s\w+요일\s-{15}"  # 예: 숫자 그룹 네 개 (4자리씩)로 된 패턴
    pattern_to_remove_name_time = r'\[.*?\]'
    for item in dialogue:
        if len(dialogue_day)>20:
            result=get_summarization(dialogue_day)
            print(result)
            if result!=None:
                summary_list.append(result)
            dialogue_day=[]
            continue
        if not re.match(pattern_to_remove_date, item):
            item = re.sub(pattern_to_remove_name_time, '', item)
            dialogue_day.append(item)
        else:
            if len(dialogue_day)!=0:
                result=get_summarization(dialogue_day)
                print(result)
                if result!=None:
                    summary_list.append(result)
                dialogue_day=[]
            else:
                continue
    
    if not summary_list :
        summary_list="We can not summarize!"
    else:
        summar_list=summary_list[:2]
    return summary_list