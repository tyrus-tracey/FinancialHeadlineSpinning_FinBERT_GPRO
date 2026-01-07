import argparse
from collections import defaultdict
import copy
import datasets as hf_datasets
import einops
import math
import numpy as np
import os
import os.path as osp
import peft
from sentence_transformers import SentenceTransformer
import random
import torch
from torch.amp import GradScaler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import wandb

import NoModify
from NoModify import SequenceRewardModel, get_system_prompt, get_data
import UtilsBase
import Utils
from Utils import twrite, unwrap_model, wrap_model, experiment_name, stateless

# Functions you need to implement; this script will not work without them!
from ImplementMe import get_per_token_logprobs, grpo_rl_objective, gpro_kl_loss, rewards_to_advantages, output_ids_to_masks

############ This *should* work automatically, but if not you can configure it #######
def get_terminal_columns(default=200):
    """Returns the number of columns in the terminal, or [default] if this does not
    work properly.
    """
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = default
    return columns
torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=get_terminal_columns())
######################################################################################
######################################################################################
######################################################################################

def get_args():
    """Returns an argparse Namespace with command-line arguments set up for use up to
    things that need to be set after training objects are constructed and DDP set up.
    """
    P = argparse.ArgumentParser()
    # Maybe interesting to try bigger models, but please use only the default 135M
    # model for parts of this assignment you submit. If you want to get more creative
    # and try others, feel free! Note that this entire codebase was written with a
    # general a padding_side='left' assumption, so using padding_side='right' models 
    # may lead to bugs.
    P.add_argument("--policy_name", default="HuggingFaceTB/SmolLM2-135M-Instruct",
        choices=["HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"],
        help="Name of the model whose weights we are fine-tuned with GRPO")

    #### Dataset arguments ###########################################################
    P.add_argument("--data_tr", default="headline2sentiment_ex32_s0_train.json",
        help="Path or specifier for the training data")
    P.add_argument("--data_val", default="headline2sentiment_ex32_s0_val.json",
        help="Path or specifier for the validation and/or test data")
    # HINT: set this very small for debugging. Once you can get good results on a
    # single data point (faster), you can increase it a bit, then to the full dataset
    P.add_argument("--data_max_ex", type=int, default=None,
        help="Maximum number of examples to use from each dataset, or None to use all examples")

    # Hyperparameters: Batch size and other how-much-parallelism arguments.
    # HINT: Set --bs to whatever you like (this primarily interacts with learning
    # rate), and then decrease --cbs till training doesn't OOM.
    P.add_argument("--bs", type=int, default=8,
        help="Number of (headline, target sentiment) examples trained on together. For every such set, we take --grpo_grad_steps gradient steps.")
    P.add_argument("--cbs", type=int, default=None,
        help="Number of completions generated from the model to process in parallel. MUST evenly divide --grpo_completions; if None, set to --grpo_completions. In general could be larger than --bs.")

    # Hyperparameters: training time and learning rate and optimizer stuff
    # As written, the starter code implements a linear-warmup-then-cosine-decay
    # learning rate schedule. The learning rate starts at zero, linearly increases to
    # its maximum value --lr over --epochs_warmup epochs, and then decays to zero for
    # the remaining epochs.
    P.add_argument("--epochs", type=int, default=32,
        help="Number of total training epochs")
    P.add_argument("--epochs_warmup", type=int, default=4,
        help="Number of training epochs to warmup learning rate for")
    
    # Optimizer hyperparameters
    P.add_argument("--lr", type=float, default=0, # HINT: tune this first XD
        help="Maximum learning rate")
    P.add_argument("--wd", type=float, default=1e-4,
        help="Weight decay for AdamW optimizer")
    P.add_argument("--beta1", type=float, default=0.95,
        help="Beta1 for AdamW optimizer")
    P.add_argument("--beta2", type=float, default=0.999,
        help="Beta2 for AdamW optimizer")

    # Hyperparameters: GRPO-specific.
    # These control the behaviour of GRPO. Currently they are set for workable but
    # not-necessarily-optimal values. Note that asking for more generated completions
    # per prompt or more tokens per generation will increase the amount of memory
    # used. It might be worth messing with the values here.
    P.add_argument("--grpo_max_new_tokens", type=int, default=32, # Decreasing this seems to hurt performance
        help="Maximum number of new tokens the model can generate for each prompt")
    P.add_argument("--grpo_completions", type=int, default=4, # More seems to be better
        help="Number of completions to sample per prompt for GRPO")
    P.add_argument("--grpo_clip", type=float, default=0.05,
        help="Clipping parameter for GRPO")
    P.add_argument("--grpo_kl_weight", type=float, default=0.04,
        help="Weight for KL penalty in GRPO")
    P.add_argument("--grpo_grad_steps", type=int, default=2,
        help="Number of gradient steps between resampling completion generations + updating pi_old")

    # Evaluation. Set to 0 to disable evaluation during training ie. when debugging
    P.add_argument("--eval_iter", type=int, default=0,
        help="Number of epochs between evaluations (0 to disable)")

    # Hardware and DDP.
    # Probably all the default arguments here are fine and should be left as is.
    # Decreases GPU memory usage and increases speed. Probably safe to leave on since I can get training to work with it
    P.add_argument("--autocast", default=True, type=lambda x: bool(int(x)),
        help="Whether to use autocasting for mixed-precision training") 
    # Multiprocessing context for DataLoader workers. 'fork' is generally fastest but sometimes unstable.
    P.add_argument("--hw_mp_ctx", choices=["fork", "spawn", "forkserver"], default="fork",
        help="Multiprocessing context for DataLoader workers")
    # Number of DataLoader workers. More is generally faster up to a point. Set to zero to diagnose DataLoader-related bugs.
    P.add_argument("--num_workers", type=int, default=4,
        help="Number of DataLoader workers")
    P.add_argument("--persistent_workers", default=True, type=lambda x: bool(int(x)),
        help="Whether DataLoader workers should be persistent. Probably faster, but could break things/deterministism")
    # Do not change this unless you're using multiple GPUs
    P.add_argument("--gpus", nargs="+", default=[0], type=int,
        help="List of GPU indices to use")
    # Single GPU model speedup options: 'gpu' is plain single-GPU, 'compile' uses
    # extra memory and time on the first bunch of training/eval steps to compile a
    # model that can be significantly faster (or break), 'lora' uses LoRA to reduce
    # the number of trainable parameters and speed up training, but possibly reduces final performance
    P.add_argument("--speedup", choices=["gpu", "lora", "compile"], default="gpu",
        help="Type of speedup to use for single-GPU training")

    # Logging and run naming and WandB
    P.add_argument("--seed", type=int, default=0,
        help="Random seed for reproducibility")
    P.add_argument("--uid", default=None,
        help="Unique identifier for this training run")
    P.add_argument("--suffix", default=None,
        help="Optional suffix to add to all logging and checkpoint files")
    P.add_argument("--tags", nargs="+", default=[],
        help="Optional list of tags to add to this run's Weights & Biases logging")
    P.add_argument("--naming", default=["default"], nargs="+",
        help="List of argument names to include in the experiment name. 'default' includes a standard set of important arguments.")
    P.add_argument("--wandb", choices=["disabled", "online", "offline"], default="disabled",
        help="Whether to use Weights & Biases logging, and if so whether to log online or offline")
    P.add_argument("--wandb_project", default="FintechCS983_2025fa_a3",
        help="Weights & Biases project name to log to")
    P.add_argument("--wandb_entity", default="your-entity", # Used to be mine, 'apex-lab', but others shouldn't use this
        help="Weights & Biases entity (team) to log to")
    
    # Saving and loading checkpoints/runs
    P.add_argument("--save_dir", default=osp.join(osp.dirname(osp.abspath(__file__)), "checkpoints"),
        help="Directory to save checkpoints and logs to")
    P.add_argument("--resume", default=None,
        help="Path to resume training from")
    P.add_argument("--save_iter", type=int, default=0,
        help="Number of epochs between saving model checkpoints (0 to disable)")
    P.add_argument("--save_iter_t", type=float, default=10,
        help="Maintain a checkpoint containing the latest model weights, updated every --save_iter_t minutes. Requires --save_iter to not be zero")
    args = P.parse_args()

    # Automically compute and set batch size arguments derived from inputs.
    args.cbs = args.cbs if args.cbs else args.grpo_completions
    args.cbs = min(args.cbs, args.grpo_completions)
    
    # Set the WandB UID
    args.uid = wandb.util.generate_id() if args.uid is None else args.uid
    return args

def get_policy_and_ref_models_tokenizer(args):
    """Returns a (model, ref_model, model_tokenizer) tuple given [args]. They have
    matching weights and use the same tokenizer.
    """
    model_tokenizer = AutoTokenizer.from_pretrained(args.policy_name,
        padding_side="left",
        use_fast=True,
        model_max_length=512)
    model_tokenizer.pad_token = model_tokenizer.eos_token if model_tokenizer.pad_token is None else model_tokenizer.pad_token
    
    # If using device_map="auto" later, use this to make HuggingFace not decide to run
    # on weird/random devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in args.gpus]) 

    model = AutoModelForCausalLM.from_pretrained(args.policy_name,
        device_map=None if "ddp" in args.speedup else "auto")
    model.generation_config.pad_token_id = model_tokenizer.pad_token_id
    
    ref_model = AutoModelForCausalLM.from_pretrained(args.policy_name, 
        device_map=None if "ddp" in args.speedup else "auto")
    ref_model.generation_config.pad_token_id = model_tokenizer.pad_token_id
    ref_model.load_state_dict(model.state_dict()) # Take no chances here

    model = model.to(Utils.get_device())
    model.train()
    ref_model = ref_model.to(Utils.get_device())
    ref_model = Utils.disable_grads(ref_model)
    ref_model.eval()
    
    return model, ref_model, model_tokenizer


@torch.no_grad()
def generate_from_batch(*, model, model_tokenizer, batch, args, device=None, seed=None, **generation_kwargs):
    """Given a batch of data [batch], returns a Namespace containing the model's
    generated outputs and related information.

    Args:
    model               -- language model
    model_tokenizer     -- tokenizer of [model]
    batch               -- batch of data as returned from DataLoaders
    args                -- Namespace with arguments
    device              -- device to run computations on
    **generation_kwargs -- additional kwargs to pass to model.generate()
    """
    device = Utils.get_device() if device is None else device
    _ = seed if seed is None else Utils.set_seed(seed)
    model.eval()

    # This shouldn't use too much VRAM, even though the completions are split into
    # --cbs chunks. If you do get an OOM, you can try writing a chunked version of
    # this, but be careful to ensure each chunk ends up with the same token-dimension
    # size; otherwise you won't be able to concatenate them to get [output_ids].
    output_ids = model.generate(
        input_ids=batch["prompt_input_ids"],
        attention_mask=batch["prompt_attention_mask"],
        # Might be interesting to come up with your own hyperparameters here!
        do_sample=True,
        temperature=1,
        max_new_tokens=args.grpo_max_new_tokens,
        num_return_sequences=args.grpo_completions,
        **generation_kwargs)

    # Compute various interesting/important binary masks
    # IMPLEMENT ME!!
    output_masks = output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=batch["prompt_input_ids"],
        pad_token_id=model_tokenizer.pad_token_id,
        tokenizer=model_tokenizer,)

    # Also decode the completions only (not the prompts)
    completions = torch.where(output_masks.completion_mask.bool(), output_ids, torch.full_like(output_ids, model_tokenizer.pad_token_id))
    completions = model_tokenizer.batch_decode(completions, skip_special_tokens=True)

    model.train()
    return argparse.Namespace(output_ids=output_ids,    # (BS C)xL tensor of token IDs, contains both prompt and response
        attention_mask=output_masks.attention_mask,     # (BS C)xL binary tensor indicating which tokens are non-pad tokens in [output_ids]
        prompt_mask=output_masks.prompt_mask,           # (BS C)xL binary tensor indicating which tokens are part of the prompt in [output_ids]
        completion_mask=output_masks.completion_mask,   # (BS C)xL binary tensor indicating which tokens are part of the response in [output_ids]
        completions=completions,)                       # (BS C)-length list of completion strings (contains only response)

@torch.no_grad()
def write_generated_sorted_by_reward(*, completions, headlines, sentiments,
    args, rewards=None, headlines_idxs=None, num_to_print=5):
    """Prints out the generated outputs sorted by their prompt and by their reward.
    
    Args:
    completions     -- (BS C)-length list of generated completion strings
    rewards     -- (BS C)-length tensor or list of sequence-level rewards
    headlines       -- BS-length list of headline strings
    sentiments      -- BS-length list of sentiment strings
    args            -- Namespace with arguments
    rewards         -- (BS C)-length tensor or list of per-completion rewards or None
    headlines_idxs  -- list of headline indices to print, or None to print all
    num_to_print    -- number of completions to print per (headline, sentiment) pair
    """
    bs_c = len(completions)
    bs = len(headlines)
    c = bs_c // bs
    assert c == args.grpo_completions, f"c={c}, expected c={args.grpo_completions}. Probably you passed in already expanded headlines/sentiments?"

    # Clean up spaces in [completions] and ensure that [rewards] is usable
    completions = [" ".join(g.split()) for g in completions]
    
    rewards = torch.ones(bs_c) if rewards is None else rewards
    rewards = rewards.cpu().squeeze().tolist() if isinstance(rewards, torch.Tensor) else rewards
    
    subsample_idxs = torch.linspace(0, c-1, min(num_to_print, c), dtype=torch.int32).tolist()
    hs2completions_rewards = dict()
    for idx in [idx for idx in range(bs) if (headlines_idxs is None or idx in headlines_idxs)]:
        completion_start_idx, completion_end_idx = idx * c, (idx+1) * c
        hs = (headlines[idx], sentiments[idx])
        completions_ = completions[completion_start_idx:completion_end_idx]
        rewards_ = rewards[completion_start_idx:completion_end_idx]

        completions_rewards = list(zip(completions_, rewards_))
        completions_rewards = sorted(completions_rewards, key=lambda x: x[1])
        completions_rewards = [completions_rewards[idx] for idx in subsample_idxs]
        hs2completions_rewards[hs] = completions_rewards

    reward_max_len = max([len(f"{r:.2f}") for v in hs2completions_rewards.values() for _,r in v])
    for (h,s), completions_rewards in hs2completions_rewards.items():
        twrite(f"=== Sentiment: {s} | Headline: {h}")
        for (comp, r) in completions_rewards:
            reward_str = f"{r:.2f}"
            twrite(f"\treward={reward_str:{reward_max_len}} | {comp}")
    twrite("")

def grpo_step_to_loss(*, output_ids, attention_mask, completion_mask, 
    logprobs_pi_old, logprobs_pi_ref,
    rewards,
    model, scaler,
    args, device, verbose=0, tokenizer=None):
    """Returns the GRPO loss for [batch].

    Args:
    output_ids          -- (BS C)x(L+1) tensor of token IDs generated from pi_old,
                            containing both prompt and response
    attention_mask      -- attention mask for [input_ids]
    logprobs_pi_old     -- (BS C)xL tensor of log-probabilities from pi_old for
                            [output_ids]. The last element in each sequence is the
                            log-probability of the pi_old model generating the
                            last token in [output_ids]
    logprobs_pi_ref     -- (BS C)xL tensor of log-probabilities from pi_ref for
                            [output_ids]. The last element in each sequence is the
                            log-probability of the pi_old model generating the last 
                            token in [output_ids]
    completion_mask     -- (BS C)x(L+1) binary tensor indicating which tokens are part
                            of the response in [output_ids]
    rewards             -- (BS C)x(L+1) tensor of rewards for [output_ids]
    model               -- the policy model being trained
    scaler              -- GradScaler for mixed-precision training
    args                -- Namespace with arguments
    device              -- device to run computations on
    verbose             -- whether to print out detailed information
    """
    # Shapes
    bs_c, seqlen = output_ids.shape
    bs = bs_c // args.grpo_completions
    assert output_ids.shape == attention_mask.shape == completion_mask.shape
    assert logprobs_pi_ref.shape == logprobs_pi_old.shape == (bs_c, seqlen-1)
    assert rewards.shape == (bs_c,)

    # Be extra extra safe and ensure no gradients are leaking through these
    logprobs_pi_old = logprobs_pi_old.detach()
    logprobs_pi_ref = logprobs_pi_ref.detach()
    rewards = rewards.detach()

    advantages = rewards_to_advantages(rewards=rewards, grpo_completions=args.grpo_completions)

    # Set masks to be boolean and compute combined mask
    completion_mask = completion_mask
    attention_mask = attention_mask
    completion_and_attention_mask = (completion_mask & attention_mask).to(advantages.dtype)

    # Remove remove the first tokens, since they correspond to the first token in the
    # prompt and hence there's no log-probability estimated for them. For
    # [advantages], we also need to change it to have a per-token shape
    completion_and_attention_mask = completion_and_attention_mask[:, 1:] 
    advantages = einops.repeat(advantages, "bsc -> bsc l", l=seqlen-1)

    # Lists we will use to accumulate losses over chunks
    loss_all, rl_loss_all, kl_loss_all = [], [], []

    # Number of chunks to break the computation into.
    num_completion_steps = math.ceil(output_ids.shape[0] / args.cbs)
    for start_idx in range(0, output_ids.shape[0], args.cbs):
        end_idx = min(start_idx + args.cbs, output_ids.shape[0])
        
        logits_model = model(input_ids=output_ids[start_idx:end_idx],
            attention_mask=attention_mask[start_idx:end_idx]).logits
        logprobs_model = get_per_token_logprobs(logits=logits_model,
            token_ids=output_ids[start_idx:end_idx])
        rl_loss = grpo_rl_objective(logprobs_model=logprobs_model,
            logprobs_pi_old=logprobs_pi_old[start_idx:end_idx],
            advantages=advantages[start_idx:end_idx],
            grpo_clip=args.grpo_clip)
        rl_loss = rl_loss * completion_and_attention_mask[start_idx:end_idx]

        kl_loss = gpro_kl_loss(logprobs_model=logprobs_model,
            logprobs_pi_ref=logprobs_pi_ref[start_idx:end_idx])
        kl_loss = kl_loss * completion_and_attention_mask[start_idx:end_idx]

        # Now compute loss, mathematically equivalent to equation (3) in the original
        # paper (arxiv.org/pdf/2402.03300), except in this case there might be more
        # than one group present. The loss should be a (BS C)xL tensor matching the
        # shape of eg. [logprobs_model].
        # HINT: read the sentences above the equation.

        loss = -(rl_loss - args.grpo_kl_weight * kl_loss)

        # With the per-token loss computed: (1) Weight the contribution of the entire
        # batch by the number of remaining tokens. This puts the mean loss per token
        # in [loss]. (3) Backpropagate the loss, and scale down by the number of
        # chunks being used since the gradients being accumulated are summed (ie. sum
        # of mean of chunks is not the same as mean of the whole)
        loss = loss.sum() / completion_and_attention_mask[start_idx:end_idx].sum().clamp(min=1.0)
        scaler.scale(loss / num_completion_steps).backward()

        loss_all.append(loss.item())
        rl_loss_all.append(rl_loss.detach().mean())
        kl_loss_all.append(kl_loss.detach().mean())

    loss = torch.tensor(loss_all).mean().item()
    rl_loss = torch.stack(rl_loss_all).mean().item()
    kl_loss = torch.stack(kl_loss_all).mean().item()

    return loss, rl_loss, kl_loss


def train_grpo_model_one_epoch(*, epoch, train_step, model, ref_model,
    model_tokenizer, sampler_tr, loader_tr, optimizer, scaler, scheduler,
    reward_model, args, device):
    """Returns a (results_epoch, model, optimizer, scaler, scheduler, loader_tr, train_step)
    tuple after training [model] for one epoch using GRPO.

    Args:
    epoch           -- current epoch number
    train_step      -- current training step number
    model           -- policy model to train
    ref_model       -- reference model for KL regularization
    model_tokenizer -- tokenizer corresponding to [model] and [ref_model]
    sampler_tr      -- DistributedSampler for training data
    loader_tr       -- DataLoader for training data
    optimizer       -- optimizer for training [model]
    scaler          -- GradScaler for mixed-precision training
    scheduler       -- learning rate scheduler for training [model]
    reward_model    -- reward model for computing sequence-level rewards
    args            -- Namespace with arguments
    device          -- device to run computations on
    """
    c = args.grpo_completions

    # Set the epoch to [sampler_tr] for proper shuffling. If we were confident
    # we'd never need to do multi-GPU training, we could construct DataLoaders in
    # a way where we don't need to do this. However, it's worthwhile to interact
    # with the code as though it were designed for large-scale training. (You can
    # ignore this line provided you don't take it out.)
    sampler_tr.set_epoch(epoch)
    model.train()

    # We log various quantities here
    grpo_idx2losses = defaultdict(lambda: 0)
    grpo_idx2rl_losses = defaultdict(lambda: 0)
    grpo_idx2kl_losses = defaultdict(lambda: 0)
    rewards_all_batches = []

    for idx,batch in tqdm(enumerate(loader_tr),
        desc=f"Batches of epoch={epoch}",
        leave=False,
        total=len(loader_tr),
        dynamic_ncols=True):
        
        # Put all the tensors on the batch onto the right device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

        # Things to log for the current batch only
        rewards_all = []
        logprobs_pi_ref_all = []
        logprobs_pi_old_all = []

        ##############################################################################
        # Draw outputs from the current model (about to be pi_old) for the ENTIRE
        # current batch. With these, compute log-probabilities and rewards for all
        # generated completions in the batch.
        ##############################################################################
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.autocast):
            with torch.no_grad():
                batch_outputs = generate_from_batch(model=model,
                    model_tokenizer=model_tokenizer, batch=batch,
                    args=args, device=device, seed=0)
                
                sentiments = [s for s in batch["sentiment"] for _ in range(args.grpo_completions)]
                headlines = [h for h in batch["headline"] for _ in range(args.grpo_completions)]
                for start_idx in range(0, batch_outputs.output_ids.shape[0], args.cbs):
                    end_idx = min(start_idx + args.cbs, batch_outputs.output_ids.shape[0])

                    # Compute log-probabilities from the reference model
                    logits_pi_ref = ref_model(input_ids=batch_outputs.output_ids[start_idx:end_idx],
                        attention_mask=batch_outputs.attention_mask[start_idx:end_idx]).logits
                    logprobs_pi_ref = get_per_token_logprobs(logits=logits_pi_ref,
                        token_ids=batch_outputs.output_ids[start_idx:end_idx]) # Implement me
                    logprobs_pi_ref_all.append(logprobs_pi_ref)
                    
                    # Compute log-probabilities from the current model being trained (pi_old)
                    logits_pi_old = model(input_ids=batch_outputs.output_ids[start_idx:end_idx],
                        attention_mask=batch_outputs.attention_mask[start_idx:end_idx]).logits
                    logprobs_pi_old = get_per_token_logprobs(logits=logits_pi_old,
                        token_ids=batch_outputs.output_ids[start_idx:end_idx]) # Implement me
                    logprobs_pi_old_all.append(logprobs_pi_old)

                    rewards = reward_model(completions=batch_outputs.completions[start_idx:end_idx],
                        sentiments=sentiments[start_idx:end_idx],
                        headlines=headlines[start_idx:end_idx])
                    rewards_all.append(rewards)

        logprobs_pi_ref = torch.cat(logprobs_pi_ref_all, dim=0).detach()
        logprobs_pi_old = torch.cat(logprobs_pi_old_all, dim=0).detach()
        rewards = torch.cat(rewards_all, dim=0)
        rewards_all_batches.append(rewards)

        # On the last batch, of an epoch, print out some completions sorted by
        # reward for monitoring purposes.
        if idx == len(loader_tr) - 1:
            _ = write_generated_sorted_by_reward(
                completions=batch_outputs.completions,
                rewards=rewards,
                headlines=batch["headline"],
                sentiments=batch["sentiment"],
                args=args)
        ##############################################################################
        ##############################################################################
        ##############################################################################

        ##############################################################################
        # Update the model using multiple GRPO steps, with each update taking a
        # gradient computed over the entire batch. Inside grpo_step_to_loss(), the
        # computation is factorized into chunks of size [args.cbs]
        ##############################################################################
        for grpo_idx in range(args.grpo_grad_steps):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.autocast):
                # When this function returns, [model] will have its gradients set for
                # having computed loss over the entire batch
                grpo_loss, rl_loss, kl_loss = grpo_step_to_loss(
                    output_ids=batch_outputs.output_ids,
                    attention_mask=batch_outputs.attention_mask,
                    logprobs_pi_ref=logprobs_pi_ref,
                    logprobs_pi_old=logprobs_pi_old,
                    completion_mask=batch_outputs.completion_mask,
                    rewards=rewards,
                    model=model,
                    scaler=scaler,
                    args=args,
                    device=device,
                    tokenizer=model_tokenizer,)

                scheduler.step(train_step)
                scaler.unscale_(optimizer)
                _ = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Do we need this??
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad(set_to_none=True)
                train_step += 1 # Each step against the gradient counts as a training step

            grpo_idx2losses[grpo_idx] += grpo_loss
            grpo_idx2kl_losses[grpo_idx] += kl_loss
            grpo_idx2rl_losses[grpo_idx] += rl_loss
    
    # Average the logged results over the batches in the epoch. Then we can compute
    # interesting stats from them to monitor training.
    reward = torch.cat(rewards_all_batches, dim=0).mean()

    grpo_idx2losses = {k: v / len(loader_tr) for k,v in grpo_idx2losses.items()}
    grpo_idx2rl_losses = {k: v / len(loader_tr) for k,v in grpo_idx2rl_losses.items()}
    grpo_idx2kl_losses = {k: v / len(loader_tr) for k,v in grpo_idx2kl_losses.items()}

    grpo_loss_start = grpo_idx2losses[0]
    grpo_loss_end = grpo_idx2losses[args.grpo_grad_steps-1]
    grpo_loss_diff = grpo_loss_start - grpo_loss_end
    grpo_loss = sum(grpo_idx2losses.values()) / len(grpo_idx2losses)

    rl_loss_start = grpo_idx2rl_losses[0]
    rl_loss_end = grpo_idx2rl_losses[args.grpo_grad_steps-1]
    rl_loss_diff = rl_loss_start - rl_loss_end
    rl_loss = sum(grpo_idx2rl_losses.values()) / len(grpo_idx2rl_losses)

    kl_loss_start = grpo_idx2kl_losses[0]
    kl_loss_end = grpo_idx2kl_losses[args.grpo_grad_steps-1]
    kl_loss_diff = kl_loss_end - kl_loss_start
    kl_loss = sum(grpo_idx2kl_losses.values()) / len(grpo_idx2kl_losses)

    results_epoch = dict(lr=scheduler.get_last_lr()[0], train_step=train_step,
        grpo_loss_start=grpo_loss_start, grpo_loss_end=grpo_loss_end,
        grpo_loss_diff=grpo_loss_diff, grpo_loss=grpo_loss,
        rl_loss_start=rl_loss_start, rl_loss_end=rl_loss_end,
        rl_loss_diff=rl_loss_diff, rl_loss=rl_loss,
        kl_loss_start=kl_loss_start, kl_loss_end=kl_loss_end,
        kl_loss_diff=kl_loss_diff, kl_loss=kl_loss,
        reward=reward)

    return results_epoch, train_step, model, optimizer, scaler, scheduler

@stateless
def eval_one_epoch(*, model, model_tokenizer, loader, reward_model, args, device, epoch=0, **kwargs):
    """Returns a results dictionary after evaluating [model] for one epoch.

    Args:
    model           -- policy model to evaluate
    model_tokenizer -- tokenizer corresponding to [model]
    loader          -- DataLoader for evaluation data
    reward_model    -- reward model for computing sequence-level rewards
    args            -- Namespace with arguments
    device          -- device to run computations on
    """
    model.eval()
    all_rewards = []
    completions = []
    headlines = []
    sentiments = []
    with torch.no_grad():
        for idx,batch in tqdm(enumerate(loader),
            desc=f"Eval batches of epoch={epoch}",
            leave=False,
            total=len(loader),
            dynamic_ncols=True):

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.autocast):
                batch_outputs = generate_from_batch(
                    model=model,
                    model_tokenizer=model_tokenizer,
                    batch=batch,
                    args=args,
                    device=device,
                )                
                # Next, compute per-sequence rewards for each completion using the reward model
                sequence_rewards = reward_model(completions=batch_outputs.completions,
                    sentiments=batch["sentiment"],
                    headlines=batch["headline"])
                all_rewards.append(sequence_rewards)
                completions += batch_outputs.completions
                headlines += batch["headline"]
                sentiments += batch["sentiment"]

    rewards = torch.cat(all_rewards, dim=0)

    twrite(f"Epoch={epoch} ====== Eval epoch generated samples sorted by reward =========")
    # Select different ones per epoch to print
    headline_idxs = [(l+epoch) % len(headlines) for l in range(len(headlines))]
    headline_idxs = headline_idxs[::max(1, len(headlines)//5)]

    _ = write_generated_sorted_by_reward(
        headlines_idxs=headline_idxs,
        completions=completions,
        rewards=rewards,
        headlines=headlines,
        sentiments=sentiments,
        args=args)

    model.train()
    results = dict(reward_mean=rewards.mean(), reward_std=rewards.std())
    return results

if __name__ == "__main__":
    args = get_args()
    _ = Utils.set_seed(args.seed, verbose=True)
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")

    ######## Get all the models we'll be using and their tokenizers ##################
    model, ref_model, model_tokenizer = get_policy_and_ref_models_tokenizer(args)
    reward_model = NoModify.SequenceRewardModel(args).to(device)
    model = Utils.wrap_model(model=model, args=args)
    ref_model = Utils.wrap_model(model=ref_model, args=args) 
    reward_model = Utils.wrap_model(model=reward_model, args=args) # Do not use LoRA here
    
    ##################################################################################
    ##################################################################################
    ##################################################################################
    # Setup: get the training and validation DataLoaders as well as the training
    # sampler. We can use sampler.set_epoch(epoch) to get nicer control over
    # data shuffling.
    sampler_tr, loader_tr = get_data(data_path=args.data_tr,
        model_tokenizer=model_tokenizer, args=args, train=True)
    _, loader_val = get_data(data_path=args.data_val,
        model_tokenizer=model_tokenizer, args=args, train=False)

    steps_per_epoch = len(loader_tr) * args.grpo_grad_steps
    warmup_steps = steps_per_epoch * args.epochs_warmup
    total_steps = steps_per_epoch * args.epochs
    args = UtilsBase.updated_namespace(args,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        total_steps=total_steps,)
    
    # Now construct optimizers, learning-rate schedulers, and gradient scalers
    optimizer = torch.optim.AdamW(unwrap_model(model).parameters(), lr=args.lr,
        betas=(0.99, 0.999), weight_decay=args.wd)
    scaler = GradScaler(enabled=bool(args.autocast))
    scheduler = Utils.LinearRampThenCosineAnnealingScheduler(optimizer,
        steps_per_epoch=args.steps_per_epoch,
        warmup_epochs=args.epochs_warmup,
        total_epochs=args.epochs,
        lr=args.lr)

    # Log training progress with these variables
    epoch, train_step, prev_time_elapsed, results = 0, 0, 0.0, dict()

    # Now that all the things we need for training, optionally resume any of them from
    # prior saved state if it exists.
    resume = Utils.args_to_latest_checkpoint(args) if args.resume == "latest" else args.resume
    if resume and osp.exists(resume):
        state = Utils.resume_all(resume_path=resume, epoch=epoch,
            train_step=train_step, results=results, hours_elapsed=prev_time_elapsed,
            model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        epoch, train_step = state["epoch"], state["train_step"]
        results, prev_time_elapsed = state["results"], state["hours_elapsed"]
        model, optimizer = state["model"], state["optimizer"]
        scaler, scheduler = state["scaler"], state["scheduler"]
        twrite(f"Resumed all states from {resume} at epoch {epoch} and train_step {train_step}")
    elif resume and not osp.exists(resume):
        raise FileNotFoundError(f"No checkpoint found at resume={resume} to resume from")
    else:
        twrite(f"No resume checkpoint specified or found. Starting training from scratch.")

    # Now we are ready to initialize WandB logging and start training. The WandB run 
    # will exist in only the main process, but other processes can pretend to log to
    # it since the receive a NoOp object as [run].
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=experiment_name(args),
        id=args.uid,
        resume="allow", # Resumes prior run if the UID matches, otherwise starts a new run
        allow_val_change=True,
        config=args,
        config_exclude_keys=["tags"],
        tags=args.tags,
        mode=args.wandb,
        dir=osp.join(args.save_dir, "wandb_logs"),)
    run.log_code(osp.dirname(__file__),
        include_fn=lambda fname: any([fname.endswith(ext) for ext in [".py", ".md"]]))

    # Print some information about the training run
    twrite(f"-------------------------------------------------------------------------")
    twrite(f"          {experiment_name(args)}")
    twrite(f"-------------------------------------------------------------------------")
    twrite(f"Args: gpus={args.gpus} bs={args.bs} cbs={args.cbs}")
    twrite(f"Args: lr={args.lr} epochs={args.epochs} epochs_warmup={args.epochs_warmup} seed={args.seed} ")
    twrite(f"Args: data={args.data_tr} data_max_ex={args.data_max_ex} -> train/val data len=({len(loader_tr.dataset)},{len(loader_val.dataset)})")
    twrite(f"Args: grpo_completions={args.grpo_completions} grpo_clip={args.grpo_clip} grpo_kl_weight={args.grpo_kl_weight}")
    twrite(f"-------------------------------------------------------------------------")    

    # Immediately prior to training, set all the seeds based on the epoch index that
    # is about to be run
    _ = Utils.set_seed(args.seed + epoch, verbose=True)
    
    start_time = time.time()
    last_save_time = time.time()
    for epoch in tqdm(range(epoch, args.epochs),
        desc="Epochs",
        dynamic_ncols=True):
        epoch_start_time = time.time()

        twrite(f"=========== Starting epoch {epoch+1}/{args.epochs} ====================")
        results[epoch] = dict()

        # Train the GRPO model [model] for one epoch. Observe in the function that
        # [train_step] has been incremented to include all the gradient steps taken
        # within the epoch, while at this point [epoch] has not yet been incremented
        results_epoch, train_step, model, optimizer, scaler, scheduler = train_grpo_model_one_epoch(
            epoch=epoch,
            train_step=train_step,
            model=model,
            ref_model=ref_model,
            model_tokenizer=model_tokenizer,
            sampler_tr=sampler_tr,
            loader_tr=loader_tr,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            reward_model=reward_model,
            args=args,
            device=device,
        )
        elapsed_time_epoch = time.time() - epoch_start_time
        results[epoch] |= results_epoch

        ###### Do an evaluation if desired, then log results #########################
        output_keys = list(results[epoch].keys())
        if args.eval_iter and ((epoch+1) % args.eval_iter == 0 or (epoch+1) == args.epochs):
            results_epoch_val = eval_one_epoch(model=model, model_tokenizer=model_tokenizer,
                reward_model=reward_model, loader=loader_val,
                args=args, device=device, epoch=epoch,)
            results_epoch_val = {f"{k}_val": v for k,v in results_epoch_val.items()}
            results[epoch] |= results_epoch_val
        
        # Add some final stats to [results]
        results[epoch] |= dict(train_step=train_step, epoch=epoch+1,
            hours_per_epoch=elapsed_time_epoch/3600,
            hours_elapsed=(time.time()-start_time)/3600 + prev_time_elapsed,)
        run.log(results[epoch], step=train_step)
        ##############################################################################
        ##############################################################################
        ##############################################################################

        ###### Print results from the current epoch ##################################
        keys_to_print = ["lr", "kl_loss", "kl_loss_diff", "grpo_loss", "reward", "reward_mean_val"]
        stats_str = " ".join([f"{k}={results[epoch][k]:.3e}" for k in keys_to_print if k in results[epoch]])
        twrite(f"Epoch {epoch+1}/{args.epochs} step={train_step}/{args.total_steps} | {stats_str}\n")

        key2results = {k: [results[epoch][k] for epoch in results if k in results[epoch]] for k in keys_to_print}
        max_key_len = max([len(k) for k in key2results.keys()])+2
        max_val_len = max([len(f"{v:.2e}") for v_list in key2results.values() for v in v_list])
        for key,results_for_key in key2results.items():
            v_list = [f"{v:.2e}" for v in results_for_key]
            twrite(f"\t{key:{max_key_len}}: " + " ".join([f"{v:{max_val_len}}" for v in v_list]))
        
        ##############################################################################
        ##############################################################################
        ##############################################################################


        ###### Save everything if desired ############################################
        save_epoch = args.save_iter and ((epoch+1) % args.save_iter == 0 or (epoch+1) == args.epochs)
        save_latest = (time.time() - last_save_time) > args.save_iter_t * 60
        if args.save_iter and (save_epoch or save_latest):
            save_latest = save_latest and not save_epoch
            _ = Utils.save_all(args=args, epoch=epoch+1, train_step=train_step,
                model=model, optimizer=optimizer, scaler=scaler,
                save_seeds=False, results=results, save_latest=save_latest,)
            last_save_time = time.time()

            # Set a seed so that we if resuming, we'd set the same seed immediately
            # prior the epoch that we're about to run. Note that this might not
            # suffice for reproducibility if using persistent DataLoader workers.
            _ = Utils.set_seed(args.seed, verbose=True)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################

        twrite(f"===================================================================")
    
    # End of training loop
    wandb.finish()
