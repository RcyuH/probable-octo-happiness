import os
import pandas as pd
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,GenerationConfig
from helper.modeling_draft import Model
from helper.get_QAs import get_test_QAs , get_train_QAs
from helper.specualtive_generate import speculative_generate
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
import signal
import sys
import torch
import random
from copy import deepcopy
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
from datetime import datetime
import argparse 
from statistics import mean , stdev
import pickle
from helper.multitask import (
    compute_multitask_reward,
    load_multitask_QAs,
    normalize_single_task_QAs,
    render_messages,
)

def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


parser = argparse.ArgumentParser(description="Training configuration")

parser.add_argument('--model_dir',type=str)
parser.add_argument('--adapter_path',type=str)
parser.add_argument('--temperature',type=float,default=1.0)
parser.add_argument('--top_p',type=float,default=0.95)
parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps for target model')
parser.add_argument('--draft_accumulation_steps', type=int, default=1, help='Gradient accumulation steps for draft model')
parser.add_argument('--target_lr', type=float, default=1e-6, help='Learning rate for target model')
parser.add_argument('--draft_lr', type=float, default=1e-4, help='Learning rate for draft model')
parser.add_argument('--is_train_draft', type=lambda x: x.lower() == 'true', default=True, help='Whether to train the draft model (True/False)')
parser.add_argument('--model_type', type=str, default='Qwen2___5-Math-7B', help='Version name for saving checkpoints')
parser.add_argument('--train_option',type=str,default="simplelr_abel_level3to5")
parser.add_argument('--task_config', type=str, default="",
                    help="Optional JSON config for multi-task RLVR datasets.")
parser.add_argument('--task_split', type=str, default="train",
                    help="Dataset split to load from task_config.")
parser.add_argument('--task_samples_per_epoch', type=int, default=None,
                    help="Optional total mixed samples per epoch for weighted task configs.")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--generation_backend', type=str, default="speculative",
                    choices=["speculative", "target"],
                    help="Use FastGRPO speculative generation or target-only generation baseline.")
parser.add_argument('--load_lora_path',type=str,default="")
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--version_name',type=str,default='normal')
parser.add_argument('--num_epochs',type=int,default=10)
parser.add_argument('--sample_num',type=int,default=100)
parser.add_argument('--grpo_iteration_num',type=int,default=1)
parser.add_argument('--repeated_generate_nums',type=int,default=8)
parser.add_argument('--beta',type=float,default=0.01)
parser.add_argument('--epsilon',type=float,default=0.1)
parser.add_argument('--max_length',type=int,default=2048)
parser.add_argument('--max_training_padding_gap',type=int,default=256)
parser.add_argument('--max_training_token',type=int,default=3072)
parser.add_argument('--log_file', type=str, required=True,
                    help="Full path to training log file, e.g., /path/to/train.log")
parser.add_argument('--saved_model_dir', type=str, required=True,
                    help="Directory to save trained target adapter/model checkpoints")
parser.add_argument('--saved_draft_model_dir', type=str, required=True,
                    help="Directory to save trained draft model checkpoints")
parser.add_argument('--saved_statistics_dir', type=str, required=True,
                    help="Directory to save statistics of generated sequence lengths.")
args = parser.parse_args()
num_epochs=args.num_epochs
sample_num=args.sample_num
grpo_iteration_num=args.grpo_iteration_num
repeated_generate_nums=args.repeated_generate_nums
beta=args.beta
epsilon=args.epsilon
max_length=args.max_length
max_training_padding_gap=args.max_training_padding_gap
max_training_token=args.max_training_token
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
draft_accumulation_steps = args.draft_accumulation_steps
target_lr = args.target_lr
draft_lr = args.draft_lr
is_train_draft = args.is_train_draft
model_type = args.model_type
model_dir = args.model_dir
adapter_path = args.adapter_path
temperature = args.temperature
top_p = args.top_p
version_name = args.version_name
log_file = args.log_file
saved_model_dir = args.saved_model_dir
saved_draft_model_dir = args.saved_draft_model_dir
saved_statistics_dir = args.saved_statistics_dir
task_config = args.task_config
task_split = args.task_split
task_samples_per_epoch = args.task_samples_per_epoch
seed = args.seed
generation_backend = args.generation_backend

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
if not os.path.exists(saved_draft_model_dir):
    os.makedirs(saved_draft_model_dir)
if not os.path.exists(saved_statistics_dir):
    os.makedirs(saved_statistics_dir)


print(datetime.now())
print(model_type,os.getenv('CUDA_VISIBLE_DEVICES'))
print("=" * 60)
print("Training & Generation Configuration")
print("=" * 60)
print(f"Model: {model_type} | Version: {version_name}")
print(f"Path: model={model_dir}, adapter={adapter_path}")
print(f"Train: epochs={num_epochs}, batch={batch_size}, "
      f"acc_steps={accumulation_steps}, draft_acc_steps={draft_accumulation_steps}")
print(f"LR: target={target_lr}, draft={draft_lr} | "
      f"Seq: max_len={max_length}, max_tokens={max_training_token}, pad_gap={max_training_padding_gap}")
print(f"Gen: temp={temperature}, top_p={top_p}"
      f"beta={beta}, epsilon={epsilon}")
print(f"Generation backend: {generation_backend}")
print(f"Task config: {task_config if task_config else args.train_option}")
print(f"Draft: train={is_train_draft}")
print(f"Iteration: grpo_iter={grpo_iteration_num}, sample={sample_num}, "
      f"repeat_gen={repeated_generate_nums}")
print("=" * 60)


config=AutoConfig.from_pretrained(model_dir)
target_model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype='auto',config=config).cuda()
target_model.eval()

config.rope_scaling=None
config.num_hidden_layers=1
model=Model(config,target_model=target_model)
model.load_model(adapter_path)
print(adapter_path)
model=model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side="left")


if config.model_type == 'llama':
    tokenizer.pad_token = "<|end_of_text|>" 
    tokenizer.pad_token_id = 128001
    

if task_config:
    QAs = load_multitask_QAs(
        task_config,
        split=task_split,
        samples_per_epoch=task_samples_per_epoch,
        seed=seed,
    )
else:
    QAs = normalize_single_task_QAs(
        get_train_QAs(args.train_option),
        task_id=args.train_option,
        prompt_type="math",
        reward_type="math_latex",
    )
print(f"Loaded {len(QAs)} training samples")
df = pd.DataFrame(QAs)

for param in model.draft_model.parameters():
    param.requires_grad=True

for param in model.target_model.parameters():
    param.requires_grad=False
for param in model.lm_head.parameters():
    param.requires_grad=False
for param in model.embed_tokens.parameters():
    param.requires_grad=False
    

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          
    r=64,                           
    lora_alpha=32,                
    lora_dropout=0.0,              
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model.target_model = get_peft_model(model.target_model,lora_config)
if  args.load_lora_path != "":
    model.target_model.load_adapter(args.load_lora_path,adapter_name="default")
model.target_model.print_trainable_parameters()

def compute_target_loss(logits,ref_logits,old_logits,labels,mask,reward,epsilon,beta,grpo_iteration):

    logits = logits[...,:-1,:].float()
    mask = mask[...,:-1]

    labels = labels.to(logits.device)
    labels = labels[..., 1:]
    
    logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if grpo_iteration==0:
        ref_logits = ref_logits[...,:-1,:].float()
        ref_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2).detach()
        old_logps = logps.clone().detach()
    else:
        ref_logps=ref_logits
        old_logps=old_logits

    coef1=torch.exp(logps-old_logps)
    coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
    loss1=torch.min(coef1*reward,coef2*reward)

    coef3=ref_logps-logps
    loss2=torch.exp(coef3)-coef3-1

    loss=-(loss1-beta*loss2)
    loss=loss*mask
    loss=loss.sum(-1)/mask.sum(-1)
    
    loss1=loss1*mask
    loss1=loss1.sum(-1)/mask.sum(-1)
    abs_loss1=torch.sum(torch.abs(loss1))
    loss2=loss2*mask
    loss2=loss2.sum(-1)/mask.sum(-1)
    
    return loss.sum(-1),abs_loss1,loss2.sum(-1),old_logps,ref_logps


def training_draft_model(model,outputs,prompt_mask):
    

    all_draft_input_states = outputs['all_draft_input_states']
    all_draft_input_ids = outputs['all_draft_input_ids']
    all_prompt_length = [prompt_mask[idx // repeated_generate_nums].sum().item() for idx in range(len(all_draft_input_states))]
    
    prompt_mask=prompt_mask.cpu()
    device=model.target_model.device
    
    sorted_pairs = sorted(
        zip(all_draft_input_ids, all_draft_input_states, all_prompt_length),
        key=lambda x: len(x[0]),
        reverse=False  
    )

    all_draft_input_ids_sorted, all_draft_input_states_sorted, all_prompt_length_sorted = zip(*sorted_pairs)

    all_draft_input_ids = list(all_draft_input_ids_sorted)
    all_draft_input_states = list(all_draft_input_states_sorted)
    all_prompt_length = list(all_prompt_length_sorted)
    
    l1_loss=torch.nn.SmoothL1Loss(reduction='none')
    total_loss1,total_loss2=0,0
    
    draft_input_states_list=[]
    draft_input_ids_list=[]
    prompt_length_list=[]
    
    cur_max_length=0
    hidden_size=all_draft_input_states[0].shape[-1]
    
    for idx , (draft_input_states,draft_input_ids,prompt_length) in enumerate(zip(all_draft_input_states,all_draft_input_ids,all_prompt_length)):
        
        if ((draft_input_ids.shape[-1]*(len(draft_input_states_list)+1)<=max_training_token*2 and
            (draft_input_ids.shape[-1]-cur_max_length)*len(draft_input_states_list)<=max_training_padding_gap) or
            len(draft_input_states_list)==0):
            
                draft_input_states_list.append(draft_input_states)
                draft_input_ids_list.append(draft_input_ids)
                prompt_length_list.append(prompt_length)
                
                cur_max_length=max(cur_max_length, draft_input_ids.shape[-1])
            
        else:
            
            cur_batch=len(draft_input_states_list)

            loss_mask=[[] for _ in range(cur_batch)]
            attention_mask=[[] for _ in range(cur_batch)]
            
            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
                attention_mask[idx_seq]=[1]*cur_len

            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                padding_len=cur_max_length-cur_len
                
                if padding_len>0:
                    draft_input_states_list[idx_seq]=torch.concat(
                        [draft_input_states_list[idx_seq],
                        torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                        dim=-2)
                    
                    draft_input_ids_list[idx_seq]=torch.concat(
                        [draft_input_ids_list[idx_seq],
                        torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                        dim=-1)
                    
                    loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
                    attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
            
            draft_input_states=torch.stack(draft_input_states_list,dim=0)
            draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
            loss_mask=torch.tensor(loss_mask,device=device)
            attention_mask=torch.tensor(attention_mask,device=device)

            with torch.amp.autocast(str(model.target_model.device),
                        dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
                draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                                attention_mask=attention_mask,use_cache=False)
                
            next_feature_states=draft_outputs['next_feature_states']
            draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
            draft_logits=model.lm_head(draft_hidden_states)
            
            with torch.no_grad():
                target_hidden_states=draft_input_states
                target_logits=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))
                target_logits=target_logits[:,1:,:].float().softmax(dim=-1).detach()
                
            loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

            loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
            loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss1=loss1.sum(-1)
            loss1=loss1*2.0
            
            draft_logits=draft_logits[:,:-1,:].float().softmax(dim=-1)

            plogp=target_logits*torch.log(draft_logits)
            loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
            loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss2= - loss2.sum(-1)

            loss2=loss2*0.1
            
            loss=loss1+loss2
                
            total_loss1+=loss1.item()
            total_loss2+=loss2.item()
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                
                loss = loss.detach()
                del loss
                torch.cuda.empty_cache()
            else:

                loss=loss/len(all_draft_input_states)
                loss=loss/draft_accumulation_steps
                loss.backward()
                
            draft_input_states_list=[all_draft_input_states[idx]]
            draft_input_ids_list=[all_draft_input_ids[idx]]
            prompt_length_list=[all_prompt_length[idx]]
            cur_max_length=all_draft_input_ids[idx].shape[-1]
            
    cur_batch=len(draft_input_states_list)

    loss_mask=[[] for _ in range(cur_batch)]
    attention_mask=[[] for _ in range(cur_batch)]
    
    cur_max_length=0
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
        attention_mask[idx_seq]=[1]*cur_len
        
        cur_max_length=max(cur_max_length, cur_len)
        
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        padding_len=cur_max_length-cur_len
        
        if padding_len>0:
            draft_input_states_list[idx_seq]=torch.concat(
                [draft_input_states_list[idx_seq],
                torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                dim=-2)
            
            draft_input_ids_list[idx_seq]=torch.concat(
                [draft_input_ids_list[idx_seq],
                torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                dim=-1)
            
            loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
            attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
    
    draft_input_states=torch.stack(draft_input_states_list,dim=0)
    draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
    loss_mask=torch.tensor(loss_mask,device=device)
    attention_mask=torch.tensor(attention_mask,device=device)
    
    with torch.amp.autocast(str(model.target_model.device),
                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                        attention_mask=attention_mask,use_cache=False)
        
    next_feature_states=draft_outputs['next_feature_states']
    draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
    draft_logits=model.lm_head(draft_hidden_states)
    
    with torch.no_grad():
        target_hidden_states=draft_input_states
        target_logits=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))
        target_logits=target_logits[:,1:,:].float().softmax(dim=-1).detach()
        
    loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

    loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
    loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss1=loss1.sum(-1)
    loss1=loss1*2.0
    
    draft_logits=draft_logits[:,:-1,:].float().softmax(dim=-1)

    plogp=target_logits*torch.log(draft_logits)
    loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
    loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss2= - loss2.sum(-1)

    loss2=loss2*0.1
    
    loss=loss1+loss2
        
    total_loss1+=loss1.item()
    total_loss2+=loss2.item()
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        
        loss = loss.detach()
        del loss
        torch.cuda.empty_cache()
    else:

        loss=loss/len(all_draft_input_states)
        loss.backward()
            
        
    total_loss1/=len(all_draft_input_states)
    total_loss2/=len(all_draft_input_states)
    
    return total_loss1,total_loss2


def target_generate(model, input_ids, attention_mask, tokenizer,
                    do_sample=False, repeated_generate_nums=None,
                    temperature=0.8, top_p=0.9, top_k=None,
                    statistical_time=True, max_length=2048):
    """Target-model generation baseline with the same output shape as speculative_generate."""
    start_time = time.time()
    device = model.target_model.device
    repeated_nums = repeated_generate_nums or 1
    prompt_length = input_ids.shape[-1]
    max_new_tokens = max(max_length - prompt_length, 1)

    expanded_input_ids = input_ids.to(device).repeat_interleave(repeated_nums, dim=0)
    expanded_attention_mask = attention_mask.to(device).repeat_interleave(repeated_nums, dim=0)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generation_kwargs = {
        "input_ids": expanded_input_ids,
        "attention_mask": expanded_attention_mask,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

    target_time_start = time.time()
    output_ids = model.target_model.generate(**generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    target_time_cost = time.time() - target_time_start

    generated_token_ids = []
    for sequence in output_ids:
        completion = sequence[prompt_length:].detach().cpu().tolist()
        trimmed = []
        for token in completion:
            if pad_token_id is not None and token == pad_token_id:
                continue
            trimmed.append(token)
            if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
                break
        generated_token_ids.append(trimmed)

    total_decoded_token_num = sum(len(item) for item in generated_token_ids)
    max_sequence_length = max((len(item) for item in generated_token_ids), default=0)
    total_time_cost = time.time() - start_time

    return {
        "generated_token_ids": generated_token_ids,
        "max_sequence_length": max_sequence_length,
        "total_acc_length": total_decoded_token_num,
        "total_acc": 1.0,
        "total_decoded_token_num": max(total_decoded_token_num, 1),
        "total_time_cost": total_time_cost,
        "target_time_cost": target_time_cost,
        "draft_time_cost": 0,
        "check_time_cost": 0,
        "prefill_time_cost": 0,
        "post_time_cost": total_time_cost - target_time_cost,
        "all_draft_input_states": None,
        "all_draft_input_ids": None,
    }


def _get_task_stat(task_stats, task_id):
    if task_id not in task_stats:
        task_stats[task_id] = {
            "used_items": 0,
            "reward_sum": 0.0,
            "reward_count": 0,
            "generated_length_sum": 0.0,
            "generated_completion_count": 0,
            "ignore_due_correct": 0,
            "ignore_due_incorrect": 0,
        }
    return task_stats[task_id]


def _summarize_task_stats(task_stats):
    summary = {}
    for task_id, stats in task_stats.items():
        reward_count = stats["reward_count"]
        completion_count = stats["generated_completion_count"]
        summary[task_id] = {
            "used_items": stats["used_items"],
            "mean_reward": round(stats["reward_sum"] / reward_count, 4) if reward_count else 0,
            "mean_length": round(stats["generated_length_sum"] / completion_count, 3) if completion_count else 0,
            "generated_completions": completion_count,
            "ignore_due_correct": stats["ignore_due_correct"],
            "ignore_due_incorrect": stats["ignore_due_incorrect"],
        }
    return summary

        
optimizer_target = torch.optim.AdamW(model.target_model.parameters(), lr=target_lr)
optimizer_draft = torch.optim.AdamW(model.draft_model.parameters(), lr=draft_lr)

with open(log_file,'w',encoding='utf-8') as f:
    pass

step=0
used_items=0
draft_step=0
draft_accumulated_step=0 
batch_logs=[]
batch_data={
    'messages':[],
    'rewards':[],
    'std_rewards':[],
    'task_ids':[],
    'generate_time_cost':0,
    'last_generate_time_cost':[],
    'train_time_cost':0,
    'last_train_time_cost':[],
    'generate_length':0,
    'last_generate_length':[],
    'total_acc_length':0,
    'last_acc_length':[],
    'total_decoded_token_num':0,
    'last_decoded_token_num':[],
    'prefill_time_cost':0,
    'target_time_cost':0,
    'draft_time_cost':0,
    'check_time_cost':0,
    'ignore_due_correct':0,
    'ignore_due_incorrect':0,
    'mean_rewards':0,
    'last_mean_rewards':[],
    'draft_train_time_cost':0,
    'last_draft_loss1':[],
    'last_draft_loss2':[] ,
    'generate_length_list':[],
    'task_stats':{},
}

optimizer_target.zero_grad(set_to_none=True)
optimizer_draft.zero_grad(set_to_none=True)
start_time=time.time()
batch=[]

class TrainDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        messages = []
        answers = []
        reward_examples = []
        task_ids = []

        for example in batch:
            messages.append(render_messages(example))
            answers.append(example.get('answer'))
            reward_examples.append(example)
            task_ids.append(example.get('task_id', 'default'))
        tokenized_inputs = self.tokenizer(
            text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
            return_tensors='pt',padding='longest',truncation=True,max_length=4096,padding_side='left'         
        )

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'messages': messages,        
            'answers': answers,
            'reward_examples': reward_examples,
            'task_ids': task_ids,
        }

dataloader=DataLoader(QAs,collate_fn=TrainDataCollator(tokenizer=tokenizer),num_workers=4,
                    persistent_workers=True,batch_size=batch_size,shuffle=True,drop_last=False)

for epoch in range(num_epochs):
    
    batch_data['ignore_due_correct']=0
    batch_data['ignore_due_incorrect']=0
    batch_data['length_stdev'] = []
    batch_data['length_range'] = []
    batch_data['length_cv'] = []
    
    for i,batch in enumerate(dataloader):
        
        if batch['input_ids'].shape[-1]>=max_length:
            batch=[]
            continue
        
        input_ids=batch['input_ids'].to('cuda')
        attention_mask=batch['attention_mask'].to('cuda')
        messages=batch['messages']
        answers=batch['answers']
        reward_examples=batch['reward_examples']
        task_ids=batch['task_ids']
        
        with torch.inference_mode():
            if generation_backend == "speculative":
                outputs=speculative_generate(model=model,input_ids=input_ids,attention_mask=attention_mask,tokenizer=tokenizer,
                do_sample=True,max_length=max_length,repeated_generate_nums=repeated_generate_nums,temperature=temperature,top_p=top_p,
                return_all_draft_input=True,statistical_time=True)
            else:
                outputs=target_generate(model=model,input_ids=input_ids,attention_mask=attention_mask,tokenizer=tokenizer,
                do_sample=True,max_length=max_length,repeated_generate_nums=repeated_generate_nums,temperature=temperature,top_p=top_p,
                statistical_time=True)
        
        prompt_length=input_ids.shape[-1]
        outputs['prompt_length']=prompt_length
        
        outputs['decoded_sequences']=[tokenizer.decode(x,skip_special_tokens=True) for x in outputs['generated_token_ids']]
        token_ids_length = [len(item) for item in outputs['generated_token_ids'] ]
        length_stdev = stdev(token_ids_length) if len(token_ids_length) > 1 else 0.0
        length_range = max(token_ids_length) - min(token_ids_length)
        length_ave = mean(token_ids_length) if len(token_ids_length) > 0 else 0
        length_cv = length_stdev / length_ave if length_ave else 0
        batch_data['generate_length_list'].extend(token_ids_length)
        
        if is_train_draft and generation_backend == "speculative":
            torch.cuda.synchronize()
            draft_train_time_start=time.time()
            draft_loss1,draft_loss2=training_draft_model(model,outputs,attention_mask)
            torch.cuda.synchronize()
            batch_data['draft_train_time_cost']+=time.time()-draft_train_time_start
            batch_data['last_draft_loss1'].append(draft_loss1)
            batch_data['last_draft_loss2'].append(draft_loss2)
            draft_accumulated_step += 1
            if is_train_draft and draft_accumulated_step % draft_accumulation_steps == 0:
                optimizer_draft.step() 
                optimizer_draft.zero_grad(set_to_none=True)
                draft_step += 1
    
        if draft_step % 1024 == 0 and step > 0 and is_train_draft and generation_backend == "speculative":
            with open(f"{saved_statistics_dir}/{step}.pkl","wb") as f:
                pickle.dump(batch_data['generate_length_list'],f)
        
        generate_length=0
        repeat_count = repeated_generate_nums or 1
        for idx_batch in range(len(answers)):
            generate_length += outputs['max_sequence_length']
            rewards=[]
            new_messages=[]
            task_id = task_ids[idx_batch]
            reward_example = reward_examples[idx_batch]
            task_stat = _get_task_stat(batch_data['task_stats'], task_id)
            cur_lengths = []

            for idx_k in range(repeat_count):
                idx_sequence=idx_batch*repeat_count+idx_k
                decoded_sequence=outputs['decoded_sequences'][idx_sequence]
                cur_lengths.append(token_ids_length[idx_sequence])
                
                new_message=deepcopy(messages[idx_batch])
                new_message.append({
                    "role": "assistant",
                    "content":decoded_sequence
                })
                
                reward=compute_multitask_reward(decoded_sequence, reward_example)
                
                rewards.append(reward)
                new_messages.append(new_message)

            task_stat['generated_length_sum'] += sum(cur_lengths)
            task_stat['generated_completion_count'] += len(cur_lengths)
            
            
            rewards=np.array(rewards) 
            if rewards.std()==0:
                
                if rewards[0]>=1.0:
                    batch_data['ignore_due_correct']+=1
                    task_stat['ignore_due_correct']+=1
                else:
                    batch_data['ignore_due_incorrect']+=1
                    task_stat['ignore_due_incorrect']+=1
                    
                continue
            
            std_rewards=(rewards-rewards.mean())/rewards.std()
            batch_data['messages']+=new_messages
            batch_data['rewards']+=rewards.tolist()
            batch_data['std_rewards']+=std_rewards.tolist()
            batch_data['task_ids'] += [task_id] * len(new_messages)
            task_stat['used_items'] += 1
            task_stat['reward_sum'] += float(rewards.sum())
            task_stat['reward_count'] += len(rewards)
            used_items+=1
            
        generate_length /= len(answers)
        
        batch_data['length_stdev'].append(length_stdev)
        batch_data['length_range'].append(length_range)
        batch_data['length_cv'].append(length_cv)
        batch_data['last_generate_time_cost'].append(outputs['total_time_cost'])
        batch_data['last_acc_length'].append(outputs['total_acc_length'])
        batch_data['last_decoded_token_num'].append(outputs['total_decoded_token_num'])
        batch_data['last_generate_length'].append(generate_length)
        batch_data['prefill_time_cost']+=outputs['prefill_time_cost']
        batch_data['target_time_cost']+=outputs['target_time_cost']
        batch_data['draft_time_cost']+=outputs['draft_time_cost']
        batch_data['check_time_cost']+=outputs['check_time_cost']
        
        batch_data['generate_time_cost']+=outputs['total_time_cost']
        batch_data['total_acc_length']+=outputs['total_acc_length']
        batch_data['total_decoded_token_num']+=outputs['total_decoded_token_num']
        batch_data['generate_length']+=generate_length
        batch=[]

        if len(batch_data['messages']) == 0:
            continue 
        
        text=tokenizer.apply_chat_template(batch_data['messages'],tokenize=False,add_generation_prompt=False)
        text=tokenizer(text,padding=False)
        loss_mask=[]
        
        for idx_message, message in enumerate(batch_data['messages']):
            prompt_text=tokenizer.apply_chat_template(message[:-1],tokenize=False,add_generation_prompt=True)
            prompt_text=tokenizer.encode(prompt_text)
            cur_loss_mask=[0]*(len(prompt_text)-1)+[1]*(len(text.input_ids[idx_message])-len(prompt_text)+1)
            loss_mask.append(cur_loss_mask)
            
        input_ids=text.input_ids
        attention_mask=text.attention_mask
        
        sorted_pairs = sorted(
            zip(input_ids, attention_mask, loss_mask, batch_data['std_rewards'], batch_data['task_ids']),
            key=lambda x: len(x[0]),
            reverse=False   
        )

        input_ids_sorted, attention_mask_sorted, loss_mask_sorted, rewards_sorted, task_ids_sorted = zip(*sorted_pairs)

        input_ids, attention_mask, loss_mask = list(input_ids_sorted), list(attention_mask_sorted), list(loss_mask_sorted)
        batch_data['std_rewards'] = list(rewards_sorted)
        batch_data['task_ids'] = list(task_ids_sorted)

        step = used_items // (batch_size * accumulation_steps)  
        batch_old_logps=[]
        batch_ref_logps=[]
        
        for grpo_iteration in range(grpo_iteration_num):
            torch.cuda.synchronize()
            train_time_start=time.time()
            
            cur_max_length=0
            device=model.target_model.device
            
            cur_input_ids=[]
            cur_attention_mask=[]
            cur_loss_mask=[]
            cur_rewards=[]
            micro_batch_index=0
            
            for j in range(len(batch_data['messages'])):
                
                if ((max(cur_max_length, len(input_ids[j])) * (len(cur_input_ids)+1)<=max_training_token and
                    (len(input_ids[j])-cur_max_length)*len(cur_input_ids)<=max_training_padding_gap) or
                    len(cur_input_ids)==0):
                    cur_max_length=max(cur_max_length, len(input_ids[j]))
                    
                    cur_input_ids.append(input_ids[j])
                    cur_attention_mask.append(attention_mask[j])
                    cur_loss_mask.append(loss_mask[j])
                    cur_rewards.append(batch_data['std_rewards'][j])
                    
                else:
                    
                    cur_batch=len(cur_input_ids)
                    for idx_seq in range(cur_batch):
                        
                        cur_len=len(cur_input_ids[idx_seq])
                        padding_len=cur_max_length-cur_len
                        
                        if padding_len>0:
                            
                            cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                            cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                            cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len
                            
                    cur_input_ids=torch.tensor(cur_input_ids, device=device)
                    cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
                    cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
                    cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

                    if grpo_iteration==0:
                        
                        model.target_model.disable_adapter_layers()
                        with torch.no_grad():
                            ref_outputs=model.target_model(cur_input_ids,cur_attention_mask)
                        ref_logits=ref_outputs.logits
                            
                    else:
                        ref_logits=batch_ref_logps[micro_batch_index]
                        
                    model.target_model.enable_adapter_layers()
                    outputs=model.target_model(cur_input_ids,cur_attention_mask)
                    
                    if grpo_iteration==0:
                        old_logits=None
                    else:
                        old_logits=batch_old_logps[micro_batch_index]
                        
                    loss,abs_loss1,loss2,old_logits,ref_logits=compute_target_loss(
                        outputs.logits,ref_logits,old_logits,
                        cur_input_ids,cur_loss_mask,cur_rewards,
                        epsilon,beta,grpo_iteration)
                        
                    if grpo_iteration==0:
                        batch_old_logps.append(old_logits)
                        batch_ref_logps.append(ref_logits)
                        
                    loss=loss/len(batch_data['messages'])
                    loss.backward()
                    micro_batch_index += 1
                    
                    cur_input_ids=[input_ids[j]]
                    cur_attention_mask=[attention_mask[j]]
                    cur_loss_mask=[loss_mask[j]]
                    cur_rewards=[batch_data['std_rewards'][j]]
                    
                    cur_max_length=len(input_ids[j])
                    
            cur_batch=len(cur_input_ids)
            for idx_seq in range(cur_batch):
                
                cur_len=len(cur_input_ids[idx_seq])
                padding_len=cur_max_length-cur_len
                
                if padding_len>0:
                    
                    cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                    cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                    cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len
                    
            cur_input_ids=torch.tensor(cur_input_ids, device=device)
            cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
            cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
            cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

            if grpo_iteration==0:
                
                model.target_model.disable_adapter_layers()
                with torch.no_grad():
                    ref_outputs=model.target_model(cur_input_ids,cur_attention_mask)
                ref_logits=ref_outputs.logits
                    
            else:
                ref_logits=batch_ref_logps[micro_batch_index]
                
            model.target_model.enable_adapter_layers()
            outputs=model.target_model(cur_input_ids,cur_attention_mask)
            
            if grpo_iteration==0:
                old_logits=None
            else:
                old_logits=batch_old_logps[micro_batch_index]
                
            loss,abs_loss1,loss2,old_logits,ref_logits=compute_target_loss(
                outputs.logits,ref_logits,old_logits,
                cur_input_ids,cur_loss_mask,cur_rewards,
                epsilon,beta,grpo_iteration)
                
            if grpo_iteration==0:
                batch_old_logps.append(old_logits)
                batch_ref_logps.append(ref_logits)
                
            loss=loss/len(batch_data['messages'])
            loss=loss/accumulation_steps
            loss.backward()

                
            optimizer_target.step()
            optimizer_target.zero_grad(set_to_none=True)
            
            torch.cuda.synchronize()
            batch_data['last_train_time_cost'].append(time.time()-train_time_start)
            batch_data['train_time_cost']+=(time.time()-train_time_start)
            batch_data['last_mean_rewards'].append(sum(batch_data['rewards'])/len(batch_data['rewards']))
            batch_data['mean_rewards']+=sum(batch_data['rewards'])/len(batch_data['rewards'])
            
            real_sample_num=sample_num*accumulation_steps
            
            avg_logs = {
                "epoch":epoch+1,
                "step": step,
                "used_items" : used_items ,
                "generation_backend":generation_backend,
                f"length_range" : round(mean(batch_data['length_range']),4),
                f"length_cv" : round(mean(batch_data['length_cv']),4) ,
                f"length_stdev" : round(mean(batch_data['length_stdev']),4) ,  
                "grpo_iteration":grpo_iteration+1,
                "used_time": round((time.time()-start_time)/60, 3),
                f"last_{sample_num}_generate_time_cost":round(sum(batch_data['last_generate_time_cost'][-real_sample_num:])/60,3),
                f"last_{sample_num}_train_time_cost": round(sum(batch_data['last_train_time_cost'][-real_sample_num:]) / 60, 3),
                f"last_{sample_num}_acc_length":round(sum(batch_data['last_acc_length'][-real_sample_num:]) / sum(batch_data['last_decoded_token_num'][-real_sample_num:]),4),
                f"last_{sample_num}_mean_rewards": round(sum(batch_data['last_mean_rewards'][-real_sample_num:]) / len(batch_data['last_mean_rewards'][-real_sample_num:]), 3),
                f"last_{sample_num}_mean_length": round(sum(batch_data['last_generate_length'][-real_sample_num:]) / len(batch_data['last_generate_length'][-real_sample_num:]), 3),
                
                "ignore_due_correct_cur_epoch":batch_data['ignore_due_correct'],
                "ignore_due_incorrect_cur_epoch":batch_data['ignore_due_incorrect'],                                
                "generate_time_cost":round(batch_data['generate_time_cost']/60,3),
                "average_acc_length":round(batch_data['total_acc_length']/batch_data['total_decoded_token_num'],4),
                "prefill_time_cost":round(batch_data['prefill_time_cost']/60,3),
                "target_time_cost":round(batch_data['target_time_cost']/60,3),
                "draft_time_cost":round(batch_data['draft_time_cost']/60,3),
                "train_time_cost":round(batch_data['train_time_cost']/60,3),
                "check_time_cost":round(batch_data['check_time_cost']/60,3),
                "mean_reward":round(batch_data['mean_rewards']/used_items,4),
                "task_metrics":_summarize_task_stats(batch_data['task_stats']),
                
                "draft_train_time_cost":round(batch_data['draft_train_time_cost']/60,3) if is_train_draft else 0, 
                f"last_{sample_num}_draft_loss1":round(sum(batch_data['last_draft_loss1'][-real_sample_num:])/len(batch_data['last_draft_loss1'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0,
                f"last_{sample_num}_draft_loss2":round(sum(batch_data['last_draft_loss2'][-real_sample_num:])/len(batch_data['last_draft_loss2'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0 
            }

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(avg_logs) + '\n')
                
            torch.cuda.empty_cache()
            
        batch_data['messages'].clear()
        batch_data['rewards'].clear()
        batch_data['std_rewards'].clear()
        batch_data['task_ids'].clear()
        batch_old_logps.clear()
        batch_ref_logps.clear()

        if step%500==0 and step!=0:
            if generation_backend == "speculative":
                model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
            model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')
            

if generation_backend == "speculative":
    model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')   
