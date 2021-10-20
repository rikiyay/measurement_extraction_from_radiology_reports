import os
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

startTime = datetime.now()

df = pd.read_csv('path_to_csv_with_reports')
idx = df.ID.tolist()
acc = df.ACC.tolist()
texts = df.findings.tolist()

pretrained = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
    
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AutoModelForQuestionAnswering.from_pretrained(pretrained)

prefix = [
    'What is the size of the',
    ]

lesion = [
    'cyst',
    'cystic lesion',
    'cystic focus',
    'low density',
    'low attenuation',
    'hypodensity',
    'hypoattenuation',
    'ipmn',
    'ipmt',
    'intraductal papillary mucinous neoplasm',
    'intraductal papillary mucinous tumor',
    'T2 bright lesion',
    'T2 bright focus',
    'T2 hyperintense lesion',
    'T2 hyperintense focus',
    ]

postfix = [
    'in the pancreas',
    'in the pancreatic head',
    'in the pancreatic neck',
    'in the pancreatic body',
    'in the pancreatic tail',
    'in the uncinate process of the pancreas',
    ]

questions = []

for i, s in enumerate(prefix):
    if i == 0:
        for j in lesion:
            for k in postfix:
                questions.append(f'{s} {j} {k}?')

results = pd.DataFrame(columns=['ID', 'ACC', 'findings']+questions)

for i, text in enumerate(texts):
    result = []
    result.append(idx[i])
    result.append(acc[i])
    result.append(texts[i])

    for j, question in enumerate(questions):
        inputs = tokenizer.encode_plus(
            question, 
            text, 
            add_special_tokens=True, 
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding='max_length',
            )
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = model(**inputs)

        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        result.append(answer)

        # print(f"Question: {question}")
        # print(f"Answer: {answer}\n")

    tmp = pd.Series(result, index=results.columns)
    results = results.append(tmp, ignore_index=True)
    if i%100 == 0:
        print(f'patient{i} done in {datetime.now()-startTime}!')

results.to_csv('output.csv', index=False)

print(datetime.now() - startTime)
