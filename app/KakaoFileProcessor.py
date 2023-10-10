'''
This file is about processing the kakao file that is uploaded as input by users.
'''

import re
import datetime 
import re
import pandas as pd

# Cutting by date - 입력된 카카오톡 대화를 날짜에 따라 자르기
def durationFilter(durationFrom,durationTo_bef,uploadedFile): 
       
       # 대화 날짜 구하기
        durationTo_bef=datetime.datetime.strptime(str(durationTo_bef), "%Y-%m-%d") #datetime으로 
        duration_from=datetime.datetime.strptime(str(durationFrom), "%Y-%m-%d")
      
        days_1=datetime.timedelta(days=1)
        durationTo=durationTo_bef+days_1
        
        date_range=pd.date_range(start=durationFrom,
                        end=durationTo)
        
        data=[]
        flag=0
        
        # 기간 따라 필터링
        for line in uploadedFile:
            date_info = "-{15}\s\d+년\s\d+월\s\d+일\s\w+요일\s-{15}" # --------------- 2021년 8월 28일 토요일 ---------------

            if re.match(date_info,line): 
                pattern="([0-9]+년)\s*([0-9]+월)\s*([0-9]+일)"
                a=re.search(pattern,line)
                if(a): #날짜 숫자만 뽑기
                    year=re.sub(r'[^0-9]', '',a.group(1))
                    month=re.sub(r'[^0-9]', '',a.group(2))
                    day=re.sub(r'[^0-9]', '',a.group(3))

                    datetime_str=f'{year}-{month}-{day}'
                    datetime_result = datetime.datetime.strptime(datetime_str, "%Y-%m-%d")
                    # 기간 전
                    if(date_range[0]>datetime_result):
                        flag=0                  
                    # 기간 내      
                    elif(date_range[0]<=datetime_result and date_range[len(date_range)-1]>datetime_result):
                        flag=1
                    # 기간 밖(더이상필요없는부분)
                    else:
                        flag=0
                        break
            if(flag==1):
                data.append(line)
                
        return data


# 연속된 발화자를 그냥 한 문장으로 만들어 버리기
def merge_conversation(uploadedFile): 
    
    names=[]
    
    for line in uploadedFile:
        date_info = "-{15}\s\d+년\s\d+월\s\d+일\s\w+요일\s-{15}" # --------------- 2021년 8월 28일 토요일 ---------------
        if re.match(date_info,line):
            continue
        if len(names)!=2:
            pattern = r"\[[^\]]+\]"
            match = re.search(pattern, line)
            if match:
                header = match.group()
                header=header.replace("[", "").replace("]", "")
                message = line.replace(header, "").strip()
                if((len(names)==1 and names[0]!=header)or len(names)==0):
                    names.append(header)

    data=[]
    
    name_before='none'
    
    for conversation in (uploadedFile): # Read data line by line
        flag=0
        
        # --------------- 2023년 4월 28일 금요일 --------------- 
        # 1. Processing the same line as above
        date_info_pattern="-{15}\s\d+년\s\d+월\s\d+일\s\w+요일\s-{15}" 
        
        if(re.match(date_info_pattern,conversation)): 
            data.append(conversation.replace("\n", " "))
            name_before='none'
            continue
        
        # 2. Processing the first conversation
        if(name_before=='none'):
            for name in names:
                if re.search(r"\[" + re.escape(name) + r"\]", conversation):
                    name_before=name 
            data.append(conversation.replace("\n", " "))
            continue
        
        # 3. Processing normal conversation
        for name in names:
            if re.search(r"\[" + re.escape(name) + r"\]", conversation):
                # Processing the consecutive speaker
                if(name==name_before):
                    pattern =r"\[.*?\] \[.*?\] (.*)"
                    match = re.sub(pattern, r"\1", conversation)
                    data[len(data)-1]=(data[len(data)-1]+str(match)).replace("\n", " ")
                else:
                    name_before=name
                    data.append(conversation.replace("\n", " "))
                flag=1
                
        # 4. If a conversation balloon contains '\n' 
        if flag==0:        
            pattern =r"\[.*?\] \[.*?\] (.*)"
            match = re.sub(pattern, r"\1", conversation)
            data[len(data)-1]=(data[len(data)-1]+str(match)).replace("\n", " ")        
            
    return data


