import fire
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed
)
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import torch
from peft import LoraConfig, TaskType, get_peft_model
import os
import re
import copy
import math
import json
import random
from al_methods import *
from utils import *


def train(
    # the only required argument
    base_model: str = "",
    dataset_name: str = "",
    output_dir: str = "",
    seed: int = 42,
    # training hyperparams
    train_batch_size: int = 8,
    test_batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    warm_up_rate: float = 0,
    gradient_accumulation_steps=1,
    input_cut_off: int = 128,
    output_cut_off: int = 256,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: List[str] = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'],
    # llm hyperparams
    do_train: bool = True,
    iter_dataset: str = None,
    al: str = "random",
):
    print("dataset_name:", dataset_name)
    print("train_batch_size:", train_batch_size)
    print("test_batch_size:", test_batch_size)
    print("output_dir:", output_dir)
    print("num_epochs:", num_epochs)
    print("learning_rate:", learning_rate)
    print("lora_r:", lora_r)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules)
    text_column = "question"
    label_column = "answer"
    set_seed(seed)

    dataset = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    task_prompt = "\nAnswer the above question. First think step by step and then answer the final number.\n"


    def prompt_process(source, target, task_prompt):
        target = target.replace("####", "The final answer is")
        return source + task_prompt + target

    def preprocess_function(examples):
        sources = examples[text_column]
        targets = examples[label_column]

        inputs = [prompt_process(source, target, task_prompt) for (source, target) in
                  zip(sources, targets)]

        model_inputs = tokenizer(inputs, max_length=input_cut_off+output_cut_off,
                                 padding="max_length", truncation=True, return_tensors='pt')

        labels = copy.deepcopy(model_inputs)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        # get the length of the target tokens. -1 to kick out the <BOS> token
        target_tokens = tokenizer(targets, padding=False)
        target_len = [len(label) - 1 for label in target_tokens['input_ids']]
        # don't calculate the loss from source and padding (left padding)
        for i in range(len(labels["input_ids"])):
            labels["input_ids"][i, :-target_len[i]] = -100

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def test_preprocess_function(examples):
        sources = examples[text_column]
        labels = examples[label_column]

        inputs = [source + task_prompt for source in sources]

        model_inputs = tokenizer(
            inputs, max_length=input_cut_off, padding="max_length", truncation=True)
        labels = tokenizer(labels, max_length=output_cut_off,
                           padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    test_dataset = processed_datasets["test"]

    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=test_batch_size, pin_memory=True
    )

    # creating model
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'lora_A' in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'lora_B' in n],
            "weight_decay": 0.5,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=learning_rate)

    # lr scheduler
    train_dataset_length = len(test_dataset)//train_batch_size + 1
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warm_up_rate*(train_dataset_length * num_epochs),
        num_training_steps=num_epochs *
        math.ceil(train_dataset_length / gradient_accumulation_steps) *
        gradient_accumulation_steps,
    )

    model, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, test_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    accs = []
    model.eval()
    instances = []
    pools = []
    best_acc = 0
    flag = 0
    if "init" in iter_dataset:
        if al == "random":
            with open(iter_dataset+"/train.jsonl", "r", encoding="utf8") as f:
                for line in f:
                    data = json.loads(line)
                    instances.append(data)
                    pools.append(data["question"])
            iter_num = 0
            os.makedirs(str(dataset_name)+"_"+al+"_iter" +
                        str(iter_num), exist_ok=True)
            with open(dataset_name + "_"+al+"_iter"+str(iter_num)+"/train.jsonl", 'w', encoding="utf-8") as file:
                for data in instances:
                    json.dump(data, file)
                    file.write('\n')
            train_dataset = load_dataset(
                "json", data_files=dataset_name + "_"+al+"_iter"+str(iter_num)+"/train.jsonl")
        else:
            with open(dataset_name+"/train.jsonl", "r", encoding="utf8") as f:
                datas = []
                for line in f:
                    data = json.loads(line)
                    datas.append(data)
                for data in tqdm(datas):
                    question = data["question"]
                    question += task_prompt
                    answer = data["answer"].replace(
                            "####", "The final answer is")
                    if "PE" in al:
                        result = PE_get_value(
                            question, answer, model, tokenizer)
                        data["PE"] = result
                    elif "Entropy" in al:
                        result = Entropy_get_value(
                            question, answer, model, tokenizer)
                        data["Entropy"] = result
                    else:
                        raise ValueError("al method error")
                    instances.append(data)
            iter_num = 0
            os.makedirs(str(dataset_name)+"_"+al+"_iter" +
                        str(iter_num), exist_ok=True)
            with open(dataset_name + "_"+al+"_iter"+str(iter_num)+"/train" + "_"+al+"_iter" + str(iter_num) + "_value_all.jsonl", 'w', encoding="utf-8") as file:
                for data in instances:
                    json.dump(data, file)
                    file.write('\n')
            with open(iter_dataset+"/train.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    pools.append(data["question"])
            now_instances = []
            for i in instances:
                if i["question"] in pools:
                    now_instances.append(i)
            instances = now_instances[::]
            with open(dataset_name + "_"+al+"_iter"+str(iter_num)+"/train" + "_"+al+"_iter" + str(iter_num) + "_value.jsonl", 'w', encoding="utf-8") as file:
                for data in now_instances:
                    json.dump(data, file)
                    file.write('\n')
            with open(dataset_name + "_"+al+"_iter"+str(iter_num)+"/train.jsonl", 'w', encoding="utf-8") as file:
                for data in now_instances:
                    json.dump(data, file)
                    file.write('\n')
            if "PE" in al:
                train_dataset = load_dataset("json", data_files=dataset_name +
                                             "_"+al+"_iter"+str(iter_num)+"/train.jsonl").remove_columns(["PE"])
            elif "Entropy" in al:
                train_dataset = load_dataset("json", data_files=dataset_name +
                                             "_"+al+"_iter"+str(iter_num)+"/train.jsonl").remove_columns(["Entropy"])
            else:
                raise ValueError("al method error")
    else:
        iter_num = int(iter_dataset[-1])
        print("Now iter num:", iter_num)
        with open(iter_dataset+"/train.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                instances.append(data)
                pools.append(data["question"])
        if "PE" in al:
            train_dataset = load_dataset(
                "json", data_files=iter_dataset+"/train.jsonl").remove_columns(["PE"])
        elif "Entropy" in al:
            train_dataset = load_dataset(
                "json", data_files=iter_dataset+"/train.jsonl").remove_columns(["Entropy"])
        elif "random" in al:
            train_dataset = load_dataset(
                "json", data_files=iter_dataset+"/train.jsonl")
        else:
            raise ValueError("al method error")
        
    with accelerator.main_process_first():
        processed_datasets = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    train_dataset = processed_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=train_batch_size, pin_memory=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    if do_train:
        for epoch in range(1, num_epochs+1):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    accelerator.print(
                        f"Epoch: {epoch} | Step: {step} | Loss: {loss}")
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

            if epoch:
                model.eval()
                test_preds = []
                for _, batch in enumerate(tqdm(test_dataloader)):
                    with torch.no_grad():
                        gen_kwargs = {
                            "max_new_tokens": output_cut_off,
                            "num_beams": 1,
                            "do_sample": False,
                        }
                        gen_kwargs["input_ids"] = batch["input_ids"]
                        gen_kwargs["attention_mask"] = batch["attention_mask"]
                        generated_tokens = accelerator.unwrap_model(
                            model).generate(**gen_kwargs, synced_gpus=is_ds_zero_3)
                    pred_tokens = generated_tokens[:, input_cut_off:]
                    pred_tokens = accelerator.pad_across_processes(
                        pred_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                    pred_tokens = accelerator.gather_for_metrics(pred_tokens)
                    test_preds.extend(tokenizer.batch_decode(
                        pred_tokens, skip_special_tokens=True))

                accelerator.wait_for_everyone()
                test_preds_cleaned = []
                for _, pred in enumerate(test_preds):
                    test_preds_cleaned.append(compute_accuracy(str(pred)))
                    test_preds[_] = str(pred).replace("\n", "")
                test_df[label_column] = test_df[label_column].apply(
                    lambda x: extract_answer_number(x.replace("####", "The final answer is")))
                assert len(test_preds_cleaned) == len(
                    test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
                test_df[label_column] = test_df[label_column].apply(
                    lambda x: x.split()[-1])
                test_df[text_column] = test_df[text_column].apply(
                    lambda x: x.replace("\n", ""))
                test_df["pred"] = test_preds_cleaned
                test_df["text_labels_orig"] = test_preds

                accelerator.print(
                    test_df[[text_column, label_column]].sample(20))
                gold = [i for i in test_df[label_column]]
                acc = compute_accuracy(test_df["pred"], gold)
                accs.append(acc)
                accelerator.print(f"{acc=}")
                os.makedirs(f"{output_dir}/iter_{iter_num+1}", exist_ok=True)
                test_df.to_csv(
                    f"{output_dir}/iter_{iter_num+1}/predictions.csv", index=False)

        temp_instances = instances[::]
        if iter_num != 9:
            now_instances = []
            if al == "random":
                with open(dataset_name+"/train.jsonl", "r", encoding="utf8") as f:
                    for line in f:
                        data = json.loads(line)
                        now_instances.append(data)
                random.shuffle(now_instances)
            elif "PE" in al:
                base_model_PE = {}
                base_model_PE_list = []
                temp_questions = []
                with open(dataset_name + "_"+al+"_iter0"+"/train" + "_"+al+"_iter0" + "_value_all.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        data["question"] = data["question"]+task_prompt
                        base_model_PE_list.append(data["PE"])
                        temp_questions.append(data["question"])
                    base_model_PE_list = normalize_list(
                        base_model_PE_list)
                    for i, j in zip(temp_questions, base_model_PE_list):
                        base_model_PE[i] = j
                with open(dataset_name+"/train.jsonl", "r", encoding="utf8") as f:
                    datas = []
                    temp_result = []
                    for line in f:
                        data = json.loads(line)
                        datas.append(data)
                    datas = [
                        i for i in datas if i["question"] not in pools]
                    for data in tqdm(datas):
                        question = data["question"]
                        if data["question"] not in pools:
                            question += task_prompt
                            answer = data["answer"].replace(
                                    "####", "The final answer is")
                            if "subtract" in al:
                                result = PE_subtract_get_value(
                                    question, answer, model, tokenizer, base_model_PE)
                            elif "dynamic" in al:
                                result = PE_dynamic_get_value(
                                    question, answer, model, tokenizer, base_model_PE, iter_num)
                            else:
                                result = PE_get_value(
                                    question, answer, model, tokenizer)
                            temp_result.append(result)
                    temp_result = normalize_list(temp_result)
                    for data, result in zip(datas, temp_result):
                        data["PE"] = result
                        now_instances.append(data)
                now_instances = sorted(
                    now_instances, key=lambda x: x['PE'], reverse=True)
            elif "Entropy" in al:
                base_model_Entropy = {}
                base_model_Entropy_list = []
                temp_questions = []
                with open(dataset_name + "_"+al+"_iter0"+"/train" + "_"+al+"_iter0" + "_value_all.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        data["question"] = data["question"]+task_prompt
                        base_model_Entropy_list.append(data["Entropy"])
                        temp_questions.append(data["question"])
                    base_model_Entropy_list = normalize_list(
                        base_model_Entropy_list)
                    for i, j in zip(temp_questions, base_model_Entropy_list):
                        base_model_Entropy[i] = j
                with open(dataset_name+"/train.jsonl", "r", encoding="utf8") as f:
                    datas = []
                    temp_data = []
                    for line in f:
                        data = json.loads(line)
                        datas.append(data)
                    datas = [
                        i for i in datas if i["question"] not in pools]
                    for data in tqdm(datas):
                        question = data["question"]
                        question += task_prompt
                        answer = data["answer"].replace(
                                    "####", "The final answer is")
                        if "subtract" in al:
                            result = Entropy_subtract_get_value(
                                question, answer, model, tokenizer, base_model_Entropy)
                        elif "dynamic" in al:
                            result = Entropy_dynamic_get_value(
                                question, answer, model, tokenizer, base_model_Entropy, iter_num)
                        else:
                            result = Entropy_get_value(
                                question, answer, model, tokenizer)
                        temp_data.append(result)
                    temp_data = normalize_list(temp_data)
                    for data,temp in zip(datas,temp_data):
                        data["Entropy"] = temp
                        now_instances.append(data)
                now_instances = sorted(
                    now_instances, key=lambda x: x['Entropy'], reverse=True)
            else:
                raise ValueError("al method error")
            if flag == 0:
                iter_num += 1
                flag = 1
            else:
                pass
            for i in now_instances:
                if i["question"] not in pools:
                    temp_instances.append(i)
                if len(temp_instances) == 500 + iter_num * 500:
                    break
            os.makedirs(f"{dataset_name}"+"_"+al+"_iter" +
                        str(iter_num), exist_ok=True)
            if "random" not in al:
                with open(dataset_name + "_"+al+"_iter"+str(iter_num)+"/train"+"_"+al+"_iter" + str(iter_num) + "_value.jsonl", 'w', encoding="utf-8") as file:
                    for data in temp_instances:
                        json.dump(data, file)
                        file.write('\n')
            with open(dataset_name + "_"+al+"_iter"+str(iter_num)+"/train.jsonl", 'w', encoding="utf-8") as file:
                for data in temp_instances:
                    json.dump(data, file)
                    file.write('\n')
        with open(f"{output_dir}/iter_{iter_num}/acc.txt", "w", encoding="utf-8") as f:
            f.write(str(accs)+"\n")

if __name__ == "__main__":
    fire.Fire(train)