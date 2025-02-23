import torch
from torch.nn.functional import softmax
import numpy as np

def PE_get_value(question, answer, model, tokenizer):
    encoded_question = tokenizer(
        question, return_tensors="pt").to(model.device)
    text = tokenizer(question+answer, return_tensors="pt").to(model.device)
    output = None
    with torch.no_grad():
        logits = model(**text)[0]
    length = len(encoded_question.input_ids[0])
    for next_token_logits, id in zip(logits[0][length-1::], text.input_ids[0][length::]):
        next_token_logits = softmax(next_token_logits, dim=-1)
        if output == None:
            output = -torch.log(next_token_logits[id]).unsqueeze(0)
        else:
            output = torch.cat(
                (output, -torch.log(next_token_logits[id]).view(1)))
    PE = torch.sum(output, dim=-1)
    return PE.item()


def PE_subtract_get_value(question, answer, model, tokenizer, base_model_PE):
    encoded_question = tokenizer(
        question, return_tensors="pt").to(model.device)
    text = tokenizer(question+answer, return_tensors="pt").to(model.device)
    output = None
    with torch.no_grad():
        logits = model(**text)[0]
    length = len(encoded_question.input_ids[0])
    for next_token_logits, id in zip(logits[0][length-1::], text.input_ids[0][length::]):
        next_token_logits = softmax(next_token_logits, dim=-1)
        if output == None:
            output = -torch.log(next_token_logits[id]).unsqueeze(0)
        else:
            output = torch.cat(
                (output, -torch.log(next_token_logits[id]).view(1)))
    PE = torch.sum(output, dim=-1).item()
    value = PE - base_model_PE[question]
    return value


def PE_dynamic_get_value(question, answer, model, tokenizer, base_model_PE, iter_num):
    PE = 0
    for i in range(3):
        encoded_question = tokenizer(
                question, return_tensors="pt").to(model.device)
        text = tokenizer(question+answer, return_tensors="pt").to(model.device)
        output = None
        logits = model(**text)[0]
        length = len(encoded_question.input_ids[0])
        for next_token_logits, id in zip(logits[0][length-1::], text.input_ids[0][length::]):
            next_token_logits = softmax(next_token_logits, dim=-1)
            if output == None:
                output = -torch.log(next_token_logits[id]).unsqueeze(0)
            else:
                output = torch.cat(
                    (output, -torch.log(next_token_logits[id]).view(1)))
        PE += torch.sum(output, dim=-1).item()
    PE = PE /3
    value = 0.1*(iter_num+1)*PE + (1-0.1*(iter_num+1))*base_model_PE[question]
    return value

def Entropy_get_value(question, answer, model, tokenizer):
    encoded_question = tokenizer(
        question, return_tensors="pt").to(model.device)
    text = tokenizer(question+answer, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**text)[0]
    length = len(encoded_question.input_ids[0])
    entropy = 0
    for next_token_logits, id in zip(logits[0][length-1::], text.input_ids[0][length::]):
        next_token_logits = softmax(next_token_logits, dim=-1)
        probs = next_token_logits.cpu().numpy()
        entropy += np.sum(-probs*np.log(probs))
    return entropy

def Entropy_subtract_get_value(question, answer, model, tokenizer, base_model_Entropy):
    encoded_question = tokenizer(
        question, return_tensors="pt").to(model.device)
    text = tokenizer(question+answer, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**text)[0]
    length = len(encoded_question.input_ids[0])
    entropy = 0
    for next_token_logits, id in zip(logits[0][length-1::], text.input_ids[0][length::]):
        next_token_logits = softmax(next_token_logits, dim=-1)
        probs = next_token_logits.cpu().numpy()
        entropy += np.sum(-probs*np.log(probs))
    value = entropy - base_model_Entropy[question]
    return value


def Entropy_dynamic_get_value(question, answer, model, tokenizer, base_model_Entropy, iter_num):
    entropy = 0
    for i in range(3):
        encoded_question = tokenizer(
            question, return_tensors="pt").to(model.device)
        text = tokenizer(question+answer, return_tensors="pt").to(model.device)
        logits = model(**text)[0]
        length = len(encoded_question.input_ids[0])
        for next_token_logits, id in zip(logits[0][length-1::], text.input_ids[0][length::]):
            next_token_logits = softmax(next_token_logits, dim=-1)
            probs = next_token_logits.detach().cpu().numpy()
            probs = np.clip(probs,a_min=1e-6,a_max=None)
            entropy += np.sum(-probs*np.log(probs))
    entropy = entropy/3
    value = 0.1*(iter_num+1)*entropy + (1-0.1*(iter_num+1))*base_model_Entropy[question]
    return value