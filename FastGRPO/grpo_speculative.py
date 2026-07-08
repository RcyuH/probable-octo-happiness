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
import math
import re
from copy import deepcopy
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
from datetime import datetime
import argparse 
from statistics import mean , stdev
import pickle
from tqdm.auto import tqdm
from helper.multitask import (
    compute_multitask_reward_debug,
    has_explicit_task_weights,
    load_multitask_QAs,
    normalize_single_task_QAs,
    render_messages,
    TaskWeightedBatchSampler,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def comma_separated_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]


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
parser.add_argument('--lora_r', type=int, default=64,
                    help="LoRA rank for target policy training.")
parser.add_argument('--lora_alpha', type=int, default=32,
                    help="LoRA alpha for target policy training.")
parser.add_argument('--lora_dropout', type=float, default=0.0,
                    help="LoRA dropout for target policy training.")
parser.add_argument('--lora_target_modules', type=comma_separated_list,
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="Comma-separated target module names for LoRA adapters.")
parser.add_argument('--lora_bias', type=str, default="none", choices=["none", "all", "lora_only"],
                    help="Bias training mode passed to PEFT LoraConfig.")
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--version_name',type=str,default='normal')
parser.add_argument('--num_epochs',type=int,default=10)
parser.add_argument('--sample_num',type=int,default=100)
parser.add_argument('--grpo_iteration_num',type=int,default=1)
parser.add_argument('--repeated_generate_nums',type=int,default=8)
parser.add_argument('--beta',type=float,default=0.01)
parser.add_argument('--epsilon',type=float,default=0.1)
parser.add_argument('--max_length',type=int,default=2048)
parser.add_argument('--verification_capacity', type=int, default=160,
                    help='Total speculative verification token budget shared across active generations.')
parser.add_argument('--max_draft_token_length', type=int, default=5,
                    help='Maximum adaptive draft-tree depth.')
parser.add_argument('--min_draft_token_length', type=int, default=3,
                    help='Minimum adaptive draft-tree depth.')
parser.add_argument('--max_draft_k', type=int, default=8,
                    help='Maximum draft branching factor.')
parser.add_argument('--max_verification_num', type=int, default=160,
                    help='Maximum verification tokens per active sequence.')
parser.add_argument('--draft_token_length_c', type=float, default=0.75,
                    help='Adaptive draft length constant; smaller values make deeper drafts.')
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
parser.add_argument('--use_tensorboard', type=str_to_bool, nargs="?", const=True, default=True,
                    help="Whether to write TensorBoard scalar logs.")
parser.add_argument('--tensorboard_log_dir', type=str, default="",
                    help="TensorBoard log directory. Defaults to a tensorboard/ folder beside --log_file.")
args = parser.parse_args()
num_epochs=args.num_epochs
sample_num=args.sample_num
grpo_iteration_num=args.grpo_iteration_num
repeated_generate_nums=args.repeated_generate_nums
beta=args.beta
epsilon=args.epsilon
max_length=args.max_length
verification_capacity = args.verification_capacity
max_draft_token_length = args.max_draft_token_length
min_draft_token_length = args.min_draft_token_length
max_draft_k = args.max_draft_k
max_verification_num = args.max_verification_num
draft_token_length_c = args.draft_token_length_c
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
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout
lora_target_modules = args.lora_target_modules
lora_bias = args.lora_bias
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
use_tensorboard = args.use_tensorboard
tensorboard_log_dir = args.tensorboard_log_dir
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if verification_capacity <= 0:
    raise ValueError("--verification_capacity must be positive.")
if batch_size <= 0:
    raise ValueError("--batch_size must be positive.")
if repeated_generate_nums <= 0:
    raise ValueError("--repeated_generate_nums must be positive.")
if max_verification_num <= 1:
    raise ValueError("--max_verification_num must be greater than 1.")
if max_draft_k <= 0:
    raise ValueError("--max_draft_k must be positive.")
if min_draft_token_length <= 0 or max_draft_token_length <= 0:
    raise ValueError("--min_draft_token_length and --max_draft_token_length must be positive.")
if min_draft_token_length > max_draft_token_length:
    raise ValueError("--min_draft_token_length must be <= --max_draft_token_length.")
if draft_token_length_c <= 0:
    raise ValueError("--draft_token_length_c must be positive.")
if generation_backend == "speculative":
    effective_generation_batch = batch_size * repeated_generate_nums
    min_verification_capacity = 2 * effective_generation_batch
    if verification_capacity < min_verification_capacity:
        raise ValueError(
            "--verification_capacity is too small for speculative generation: "
            f"got {verification_capacity}, but --batch_size {batch_size} * "
            f"--repeated_generate_nums {repeated_generate_nums} requires at least "
            f"{min_verification_capacity} verification slots. Increase "
            "--verification_capacity or reduce --batch_size/--repeated_generate_nums."
        )
if lora_r <= 0:
    raise ValueError("--lora_r must be positive.")
if lora_alpha <= 0:
    raise ValueError("--lora_alpha must be positive.")
if not 0 <= lora_dropout < 1:
    raise ValueError("--lora_dropout must be in [0, 1).")
if not lora_target_modules:
    raise ValueError("--lora_target_modules must contain at least one module name.")

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
print("LoRA: "
      f"r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, "
      f"bias={lora_bias}, target_modules={','.join(lora_target_modules)}")
print(f"Gen: temp={temperature}, top_p={top_p}"
      f"beta={beta}, epsilon={epsilon}")
print("Speculative: "
      f"verification_capacity={verification_capacity}, max_verification_num={max_verification_num}, "
      f"min_draft_len={min_draft_token_length}, max_draft_len={max_draft_token_length}, "
      f"max_draft_k={max_draft_k}, draft_len_c={draft_token_length_c}")
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
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    bias=lora_bias,
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
        "speculative_emitted_tokens": total_decoded_token_num,
        "speculative_accepted_draft_tokens": 0,
        "speculative_verified_draft_tokens": 0,
        "speculative_path_budget_tokens": 0,
        "speculative_verification_rounds": 0,
        "total_time_cost": total_time_cost,
        "target_time_cost": target_time_cost,
        "draft_time_cost": 0,
        "check_time_cost": 0,
        "prefill_time_cost": 0,
        "post_time_cost": total_time_cost - target_time_cost,
        "all_draft_input_states": None,
        "all_draft_input_ids": None,
    }


def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _generation_perf_metrics(outputs, token_ids_length):
    generated_completion_tokens = int(sum(token_ids_length))
    total_time = float(outputs.get("total_time_cost", 0.0) or 0.0)
    emitted_tokens = int(outputs.get("speculative_emitted_tokens", outputs.get("total_acc_length", 0)) or 0)
    accepted_draft_tokens = int(outputs.get("speculative_accepted_draft_tokens", 0) or 0)
    verified_draft_tokens = int(outputs.get("speculative_verified_draft_tokens", 0) or 0)
    path_budget_tokens = int(outputs.get("speculative_path_budget_tokens", 0) or 0)
    verification_rounds = int(outputs.get("speculative_verification_rounds", outputs.get("total_decoded_token_num", 0)) or 0)

    return {
        "generated_completion_tokens": generated_completion_tokens,
        "generated_tokens_per_second": round(_safe_div(generated_completion_tokens, total_time), 4),
        "speculative_verification_rounds": verification_rounds,
        "speculative_emitted_tokens": emitted_tokens,
        "speculative_accepted_draft_tokens": accepted_draft_tokens,
        "speculative_verified_draft_tokens": verified_draft_tokens,
        "speculative_path_budget_tokens": path_budget_tokens,
        "speculative_avg_emitted_tokens_per_round": round(_safe_div(emitted_tokens, verification_rounds), 4),
        "speculative_avg_accepted_draft_tokens_per_round": round(_safe_div(accepted_draft_tokens, verification_rounds), 4),
        "speculative_path_acceptance_rate": round(_safe_div(accepted_draft_tokens, path_budget_tokens), 6),
        "speculative_tree_acceptance_rate": round(_safe_div(accepted_draft_tokens, verified_draft_tokens), 6),
        "speculative_verified_draft_tokens_per_round": round(_safe_div(verified_draft_tokens, verification_rounds), 4),
        "target_time_ratio": round(_safe_div(outputs.get("target_time_cost", 0.0), total_time), 6),
        "draft_time_ratio": round(_safe_div(outputs.get("draft_time_cost", 0.0), total_time), 6),
        "check_time_ratio": round(_safe_div(outputs.get("check_time_cost", 0.0), total_time), 6),
    }


def _new_reward_debug_stats():
    return {
        "completion_count": 0,
        "reward_sum": 0.0,
        "reward_sq_sum": 0.0,
        "pass_count": 0,
        "fail_count": 0,
        "timeout_count": 0,
        "missing_tests_count": 0,
        "missing_entry_point_count": 0,
        "completion_chars_sum": 0.0,
        "extracted_code_chars_sum": 0.0,
        "stdout_chars_sum": 0.0,
        "stderr_chars_sum": 0.0,
        "used_group_count": 0,
        "skip_group_count": 0,
        "skip_due_correct_group_count": 0,
        "skip_due_incorrect_group_count": 0,
        "reward_type_counts": {},
        "error_type_counts": {},
        "test_type_counts": {},
        "ignored_correct_error_type_counts": {},
        "ignored_incorrect_error_type_counts": {},
    }


def _increment_counter(counter, key, amount=1):
    key = str(key if key not in (None, "") else "unknown")
    counter[key] = counter.get(key, 0) + amount


def _record_reward_detail(stats, detail):
    reward = float(detail.get("reward", 0.0))
    stats["completion_count"] += 1
    stats["reward_sum"] += reward
    stats["reward_sq_sum"] += reward * reward
    if detail.get("passed") or reward >= 1.0:
        stats["pass_count"] += 1
    else:
        stats["fail_count"] += 1
    if detail.get("timed_out"):
        stats["timeout_count"] += 1
    if detail.get("has_tests") is False:
        stats["missing_tests_count"] += 1
    test_type = str(detail.get("test_type") or "")
    if (detail.get("has_entry_point") is False and
            detail.get("reward_type") == "code_unit_test" and
            test_type not in ("stdin_stdout", "stdio", "io", "input_output")):
        stats["missing_entry_point_count"] += 1

    for field in ("completion_chars", "extracted_code_chars", "stdout_chars", "stderr_chars"):
        if field in detail:
            stats[f"{field}_sum"] += float(detail.get(field) or 0.0)

    _increment_counter(stats["reward_type_counts"], detail.get("reward_type"))
    _increment_counter(stats["error_type_counts"], detail.get("error_type"))
    if detail.get("test_type") is not None:
        _increment_counter(stats["test_type_counts"], detail.get("test_type"))


def _record_group_decision(stats, reward_details, decision):
    if decision == "used":
        stats["used_group_count"] += 1
        return

    stats["skip_group_count"] += 1
    if decision == "ignore_due_correct":
        stats["skip_due_correct_group_count"] += 1
        target_counter = stats["ignored_correct_error_type_counts"]
    else:
        stats["skip_due_incorrect_group_count"] += 1
        target_counter = stats["ignored_incorrect_error_type_counts"]
    for detail in reward_details:
        _increment_counter(target_counter, detail.get("error_type"))


def _summarize_reward_debug(stats):
    completion_count = stats["completion_count"]
    reward_mean = stats["reward_sum"] / completion_count if completion_count else 0.0
    reward_var = stats["reward_sq_sum"] / completion_count - reward_mean * reward_mean if completion_count else 0.0
    reward_var = max(reward_var, 0.0)
    return {
        "completion_count": completion_count,
        "mean_reward_all_completions": round(reward_mean, 4),
        "reward_std_all_completions": round(math.sqrt(reward_var), 4),
        "pass_rate": round(stats["pass_count"] / completion_count, 4) if completion_count else 0,
        "fail_rate": round(stats["fail_count"] / completion_count, 4) if completion_count else 0,
        "timeout_count": stats["timeout_count"],
        "missing_tests_count": stats["missing_tests_count"],
        "missing_entry_point_count": stats["missing_entry_point_count"],
        "mean_completion_chars": round(stats["completion_chars_sum"] / completion_count, 2) if completion_count else 0,
        "mean_extracted_code_chars": round(stats["extracted_code_chars_sum"] / completion_count, 2) if completion_count else 0,
        "used_group_count": stats["used_group_count"],
        "skip_group_count": stats["skip_group_count"],
        "skip_due_correct_group_count": stats["skip_due_correct_group_count"],
        "skip_due_incorrect_group_count": stats["skip_due_incorrect_group_count"],
        "reward_type_counts": dict(stats["reward_type_counts"]),
        "error_type_counts": dict(stats["error_type_counts"]),
        "test_type_counts": dict(stats["test_type_counts"]),
        "ignored_correct_error_type_counts": dict(stats["ignored_correct_error_type_counts"]),
        "ignored_incorrect_error_type_counts": dict(stats["ignored_incorrect_error_type_counts"]),
    }


def _sanitize_tb_tag(value):
    return re.sub(r"[^A-Za-z0-9_./-]", "_", str(value)).strip("/") or "unknown"


_TB_TOP_LEVEL_KEYS = {
    "generation": (
        "used_items",
        "generated_group_count",
        "batch_prompt_count",
        "batch_completion_count",
        "batch_used_group_count",
        "batch_ignore_due_correct",
        "batch_ignore_due_incorrect",
        "batch_generate_time_cost",
        "batch_mean_length",
        "batch_length_stdev",
        "batch_length_range",
        "batch_length_cv",
        "batch_average_acc_length",
    ),
    "train": (
        "used_items",
        "generated_group_count",
        "used_time",
        "length_range",
        "length_cv",
        "length_stdev",
        "ignore_due_correct_cur_epoch",
        "ignore_due_incorrect_cur_epoch",
        "generate_time_cost",
        "train_time_cost",
        "average_acc_length",
        "mean_reward",
        "draft_train_time_cost",
    ),
}
_TB_LAST_KEY_MARKERS = (
    "generate_time_cost",
    "train_time_cost",
    "acc_length",
    "mean_rewards",
    "mean_length",
    "draft_loss1",
    "draft_loss2",
)
_TB_GENERATION_PERF_KEYS = (
    "generated_tokens_per_second",
    "speculative_verification_rounds",
    "speculative_avg_emitted_tokens_per_round",
    "speculative_avg_accepted_draft_tokens_per_round",
    "speculative_path_acceptance_rate",
    "speculative_tree_acceptance_rate",
    "speculative_verified_draft_tokens_per_round",
)
_TB_REWARD_DEBUG_KEYS = (
    "completion_count",
    "mean_reward_all_completions",
    "reward_std_all_completions",
    "pass_rate",
    "fail_rate",
    "timeout_count",
    "missing_tests_count",
    "missing_entry_point_count",
    "used_group_count",
    "skip_group_count",
    "skip_due_correct_group_count",
    "skip_due_incorrect_group_count",
)
_TB_TASK_KEYS = (
    "used_items",
    "mean_reward",
    "mean_reward_all_completions",
    "mean_length",
    "generated_completions",
    "ignore_due_correct",
    "ignore_due_incorrect",
)


def _add_tb_scalar(writer, tag, value, step):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            writer.add_scalar(tag, value, step)


def _write_named_scalars(writer, values, keys, step, prefix):
    for key in keys:
        if key in values:
            _add_tb_scalar(writer, f"{prefix}/{_sanitize_tb_tag(key)}", values[key], step)


def _write_tensorboard_scalars(writer, payload, step, prefix):
    if writer is None:
        return
    event = payload.get("event", prefix)
    base_tag = _sanitize_tb_tag(prefix)
    _write_named_scalars(writer, payload, _TB_TOP_LEVEL_KEYS.get(event, ()), step, base_tag)

    if event == "train":
        for key, value in payload.items():
            if key.startswith("last_") and any(marker in key for marker in _TB_LAST_KEY_MARKERS):
                _add_tb_scalar(writer, f"{base_tag}/{_sanitize_tb_tag(key)}", value, step)

    generation_perf = payload.get("generation_perf")
    if isinstance(generation_perf, dict):
        _write_named_scalars(
            writer,
            generation_perf,
            _TB_GENERATION_PERF_KEYS,
            step,
            f"{base_tag}/generation_perf",
        )

    for reward_key in ("reward_debug_batch", "reward_debug"):
        reward_debug = payload.get(reward_key)
        if isinstance(reward_debug, dict):
            _write_named_scalars(
                writer,
                reward_debug,
                _TB_REWARD_DEBUG_KEYS,
                step,
                f"{base_tag}/{_sanitize_tb_tag(reward_key)}",
            )

    task_metrics = payload.get("task_metrics")
    if isinstance(task_metrics, dict):
        for task_id, metrics in task_metrics.items():
            if not isinstance(metrics, dict):
                continue
            task_tag = f"{base_tag}/task/{_sanitize_tb_tag(task_id)}"
            _write_named_scalars(writer, metrics, _TB_TASK_KEYS, step, task_tag)
            reward_debug = metrics.get("reward_debug")
            if isinstance(reward_debug, dict):
                _write_named_scalars(
                    writer,
                    reward_debug,
                    _TB_REWARD_DEBUG_KEYS,
                    step,
                    f"{task_tag}/reward_debug",
                )
    writer.flush()


def _write_log_event(log_file, writer, payload, tb_step, tb_prefix):
    payload["tb_step"] = tb_step
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload) + '\n')
    _write_tensorboard_scalars(writer, payload, tb_step, tb_prefix)


def _build_tensorboard_writer(use_tensorboard, tensorboard_log_dir, log_file):
    if not use_tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard logging requested, but tensorboard is not installed. Continuing with JSONL logs only.")
        return None
    if not tensorboard_log_dir:
        base_log_dir = os.path.dirname(log_file) or "."
        tensorboard_log_dir = os.path.join(base_log_dir, "tensorboard")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    print(f"TensorBoard log dir: {tensorboard_log_dir}")
    return SummaryWriter(log_dir=tensorboard_log_dir)


def _get_task_stat(task_stats, task_id):
    if task_id not in task_stats:
        task_stats[task_id] = {
            "used_items": 0,
            "reward_sum": 0.0,
            "reward_count": 0,
            "all_reward_sum": 0.0,
            "all_reward_count": 0,
            "generated_length_sum": 0.0,
            "generated_completion_count": 0,
            "ignore_due_correct": 0,
            "ignore_due_incorrect": 0,
            "reward_debug": _new_reward_debug_stats(),
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
            "mean_reward_all_completions": round(stats["all_reward_sum"] / stats["all_reward_count"], 4) if stats["all_reward_count"] else 0,
            "mean_length": round(stats["generated_length_sum"] / completion_count, 3) if completion_count else 0,
            "generated_completions": completion_count,
            "ignore_due_correct": stats["ignore_due_correct"],
            "ignore_due_incorrect": stats["ignore_due_incorrect"],
            "reward_debug": _summarize_reward_debug(stats["reward_debug"]),
        }
    return summary

        
optimizer_target = torch.optim.AdamW(model.target_model.parameters(), lr=target_lr)
optimizer_draft = torch.optim.AdamW(model.draft_model.parameters(), lr=draft_lr)

log_dir = os.path.dirname(log_file)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
with open(log_file,'w',encoding='utf-8') as f:
    pass
tb_writer = _build_tensorboard_writer(use_tensorboard, tensorboard_log_dir, log_file)
tb_log_step = 0

step=0
used_items=0
generated_group_count=0
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
    'generated_completion_tokens':0,
    'speculative_emitted_tokens':0,
    'speculative_accepted_draft_tokens':0,
    'speculative_verified_draft_tokens':0,
    'speculative_path_budget_tokens':0,
    'speculative_verification_rounds':0,
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
    'reward_debug':_new_reward_debug_stats(),
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

if has_explicit_task_weights(QAs):
    task_batch_sampler = TaskWeightedBatchSampler(
        QAs,
        batch_size=batch_size,
        seed=seed,
        drop_last=False,
    )
    print(f"Using task-weighted batch sampler: {task_batch_sampler.batch_task_counts()}")
    dataloader=DataLoader(QAs,collate_fn=TrainDataCollator(tokenizer=tokenizer),num_workers=4,
                        persistent_workers=True,batch_sampler=task_batch_sampler)
else:
    dataloader=DataLoader(QAs,collate_fn=TrainDataCollator(tokenizer=tokenizer),num_workers=4,
                        persistent_workers=True,batch_size=batch_size,shuffle=True,drop_last=False)

train_progress = tqdm(
    total=num_epochs * len(dataloader),
    desc="Training",
    dynamic_ncols=True,
    unit="batch",
)

for epoch in range(num_epochs):
    
    batch_data['ignore_due_correct']=0
    batch_data['ignore_due_incorrect']=0
    batch_data['length_stdev'] = []
    batch_data['length_range'] = []
    batch_data['length_cv'] = []
    
    for i,batch in enumerate(dataloader):
        train_progress.set_description(f"Epoch {epoch+1}/{num_epochs}")
        
        if batch['input_ids'].shape[-1]>=max_length:
            batch=[]
            train_progress.set_postfix({
                "used": used_items,
                "skip": "prompt>=max_length",
            })
            train_progress.update(1)
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
                verification_capacity=verification_capacity,max_draft_token_length=max_draft_token_length,
                min_draft_token_length=min_draft_token_length,max_draft_k=max_draft_k,
                max_verification_num=max_verification_num,draft_token_length_c=draft_token_length_c,
                return_all_draft_input=True,statistical_time=True)
            else:
                outputs=target_generate(model=model,input_ids=input_ids,attention_mask=attention_mask,tokenizer=tokenizer,
                do_sample=True,max_length=max_length,repeated_generate_nums=repeated_generate_nums,temperature=temperature,top_p=top_p,
                statistical_time=True)
        
        prompt_length=input_ids.shape[-1]
        outputs['prompt_length']=prompt_length
        
        outputs['decoded_sequences']=[tokenizer.decode(x,skip_special_tokens=True) for x in outputs['generated_token_ids']]
        token_ids_length = [len(item) for item in outputs['generated_token_ids'] ]
        generation_perf = _generation_perf_metrics(outputs, token_ids_length)
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
        batch_reward_debug = _new_reward_debug_stats()
        batch_used_groups = 0
        batch_skip_due_correct = 0
        batch_skip_due_incorrect = 0
        for idx_batch in range(len(answers)):
            generate_length += outputs['max_sequence_length']
            rewards=[]
            reward_details=[]
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
                
                reward_detail=compute_multitask_reward_debug(decoded_sequence, reward_example)
                reward=float(reward_detail["reward"])
                
                rewards.append(reward)
                reward_details.append(reward_detail)
                new_messages.append(new_message)
                _record_reward_detail(batch_reward_debug, reward_detail)
                _record_reward_detail(batch_data['reward_debug'], reward_detail)
                _record_reward_detail(task_stat['reward_debug'], reward_detail)

            task_stat['generated_length_sum'] += sum(cur_lengths)
            task_stat['generated_completion_count'] += len(cur_lengths)
            task_stat['all_reward_sum'] += float(sum(rewards))
            task_stat['all_reward_count'] += len(rewards)
            generated_group_count += 1
            
            
            rewards=np.array(rewards) 
            if rewards.std()==0:
                
                if rewards[0]>=1.0:
                    batch_data['ignore_due_correct']+=1
                    task_stat['ignore_due_correct']+=1
                    batch_skip_due_correct += 1
                    _record_group_decision(batch_reward_debug, reward_details, "ignore_due_correct")
                    _record_group_decision(batch_data['reward_debug'], reward_details, "ignore_due_correct")
                    _record_group_decision(task_stat['reward_debug'], reward_details, "ignore_due_correct")
                else:
                    batch_data['ignore_due_incorrect']+=1
                    task_stat['ignore_due_incorrect']+=1
                    batch_skip_due_incorrect += 1
                    _record_group_decision(batch_reward_debug, reward_details, "ignore_due_incorrect")
                    _record_group_decision(batch_data['reward_debug'], reward_details, "ignore_due_incorrect")
                    _record_group_decision(task_stat['reward_debug'], reward_details, "ignore_due_incorrect")
                    
                continue
            
            std_rewards=(rewards-rewards.mean())/rewards.std()
            batch_data['messages']+=new_messages
            batch_data['rewards']+=rewards.tolist()
            batch_data['std_rewards']+=std_rewards.tolist()
            batch_data['task_ids'] += [task_id] * len(new_messages)
            task_stat['used_items'] += 1
            task_stat['reward_sum'] += float(rewards.sum())
            task_stat['reward_count'] += len(rewards)
            batch_used_groups += 1
            _record_group_decision(batch_reward_debug, reward_details, "used")
            _record_group_decision(batch_data['reward_debug'], reward_details, "used")
            _record_group_decision(task_stat['reward_debug'], reward_details, "used")
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
        batch_data['generated_completion_tokens']+=generation_perf['generated_completion_tokens']
        batch_data['speculative_emitted_tokens']+=generation_perf['speculative_emitted_tokens']
        batch_data['speculative_accepted_draft_tokens']+=generation_perf['speculative_accepted_draft_tokens']
        batch_data['speculative_verified_draft_tokens']+=generation_perf['speculative_verified_draft_tokens']
        batch_data['speculative_path_budget_tokens']+=generation_perf['speculative_path_budget_tokens']
        batch_data['speculative_verification_rounds']+=generation_perf['speculative_verification_rounds']
        batch_data['generate_length']+=generate_length
        generation_logs = {
            "event": "generation",
            "epoch": epoch+1,
            "batch_index": i,
            "step": step,
            "used_items": used_items,
            "generated_group_count": generated_group_count,
            "generation_backend": generation_backend,
            "batch_prompt_count": len(answers),
            "batch_completion_count": len(outputs['generated_token_ids']),
            "batch_used_group_count": batch_used_groups,
            "batch_ignore_due_correct": batch_skip_due_correct,
            "batch_ignore_due_incorrect": batch_skip_due_incorrect,
            "ignore_due_correct_cur_epoch": batch_data['ignore_due_correct'],
            "ignore_due_incorrect_cur_epoch": batch_data['ignore_due_incorrect'],
            "batch_generate_time_cost": round(outputs['total_time_cost'], 4),
            "batch_mean_length": round(generate_length, 3),
            "batch_length_stdev": round(length_stdev, 4),
            "batch_length_range": length_range,
            "batch_length_cv": round(length_cv, 4),
            "batch_average_acc_length": round(outputs['total_acc_length'] / outputs['total_decoded_token_num'], 4) if outputs['total_decoded_token_num'] else 0,
            "generation_perf": generation_perf,
            "reward_debug_batch": _summarize_reward_debug(batch_reward_debug),
            "reward_debug": _summarize_reward_debug(batch_data['reward_debug']),
            "task_metrics": _summarize_task_stats(batch_data['task_stats']),
        }
        _write_log_event(log_file, tb_writer, generation_logs, tb_log_step, "generation")
        tb_log_step += 1
        progress_postfix = {
            "used": used_items,
            "reward": generation_logs["reward_debug_batch"]["mean_reward_all_completions"],
            "tok/s": generation_perf["generated_tokens_per_second"],
            "skip": batch_skip_due_correct + batch_skip_due_incorrect,
        }
        batch=[]

        if len(batch_data['messages']) == 0:
            train_progress.set_postfix(progress_postfix)
            train_progress.update(1)
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
            cumulative_generation_perf = {
                "generated_completion_tokens": batch_data['generated_completion_tokens'],
                "generated_tokens_per_second": round(_safe_div(batch_data['generated_completion_tokens'], batch_data['generate_time_cost']), 4),
                "speculative_verification_rounds": batch_data['speculative_verification_rounds'],
                "speculative_emitted_tokens": batch_data['speculative_emitted_tokens'],
                "speculative_accepted_draft_tokens": batch_data['speculative_accepted_draft_tokens'],
                "speculative_verified_draft_tokens": batch_data['speculative_verified_draft_tokens'],
                "speculative_path_budget_tokens": batch_data['speculative_path_budget_tokens'],
                "speculative_avg_emitted_tokens_per_round": round(_safe_div(batch_data['speculative_emitted_tokens'], batch_data['speculative_verification_rounds']), 4),
                "speculative_avg_accepted_draft_tokens_per_round": round(_safe_div(batch_data['speculative_accepted_draft_tokens'], batch_data['speculative_verification_rounds']), 4),
                "speculative_path_acceptance_rate": round(_safe_div(batch_data['speculative_accepted_draft_tokens'], batch_data['speculative_path_budget_tokens']), 6),
                "speculative_tree_acceptance_rate": round(_safe_div(batch_data['speculative_accepted_draft_tokens'], batch_data['speculative_verified_draft_tokens']), 6),
                "speculative_verified_draft_tokens_per_round": round(_safe_div(batch_data['speculative_verified_draft_tokens'], batch_data['speculative_verification_rounds']), 4),
            }
            
            avg_logs = {
                "event":"train",
                "epoch":epoch+1,
                "step": step,
                "used_items" : used_items ,
                "generated_group_count": generated_group_count,
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
                "mean_reward":round(batch_data['mean_rewards']/used_items,4) if used_items else 0,
                "generation_perf": cumulative_generation_perf,
                "reward_debug":_summarize_reward_debug(batch_data['reward_debug']),
                "task_metrics":_summarize_task_stats(batch_data['task_stats']),
                
                "draft_train_time_cost":round(batch_data['draft_train_time_cost']/60,3) if is_train_draft else 0, 
                f"last_{sample_num}_draft_loss1":round(sum(batch_data['last_draft_loss1'][-real_sample_num:])/len(batch_data['last_draft_loss1'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0,
                f"last_{sample_num}_draft_loss2":round(sum(batch_data['last_draft_loss2'][-real_sample_num:])/len(batch_data['last_draft_loss2'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0 
            }

            _write_log_event(log_file, tb_writer, avg_logs, tb_log_step, "train")
            tb_log_step += 1
            progress_postfix.update({
                "reward": avg_logs["mean_reward"],
                "train_min": avg_logs["train_time_cost"],
            })
                
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
            
        train_progress.set_postfix(progress_postfix)
        train_progress.update(1)


train_progress.close()

if generation_backend == "speculative":
    model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')   
if tb_writer is not None:
    tb_writer.close()
