import argparse
import einops
from functools import partial
import json
import os
import os.path as osp
import torch
import torch.nn as nn
from tqdm import tqdm
import traceback

import UtilsBase
from UtilsBase import to_json_encoding, from_json_encoding  # Use to encode/decode tensors in test cases
from Utils import twrite                                    # Useful for debugging

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

def output_ids_to_masks(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    device = output_ids.device
    BSxC,T = output_ids.shape
    BS, S = prompt_input_ids.shape
    C = int(BSxC / BS)

    attention_mask = torch.where(output_ids != pad_token_id, 1, 0).to(device)

    # Calculate length of prompt i:
    #   S - starting index of prompt i
    input_prompt_start= torch.argmax(
        (prompt_input_ids != pad_token_id).long(),
        dim=1
    ).to(device)

    prompt_lengths = torch.sub(
        torch.full([BS], S).to(device),
        input_prompt_start
    ).to(device)

    C_prompt_lengths = einops.repeat(prompt_lengths, "BS -> (BS c)", c=C).to(device)

    # prompt region = output_prompt_start + prompt_length
    # Note: for each input prompt length need to apply to C
    #   rows in output_prompt_start

    output_ids_indices_per_row = einops.repeat(torch.arange(0, T), "T -> (bsxc) T", bsxc=BSxC).to(device)
    output_prompt_start = torch.argmax(
        (output_ids != pad_token_id).long(),
        dim=1
    ).to(device)
    
    output_prompt_end = torch.add(output_prompt_start, C_prompt_lengths)
    output_prompt_start = output_prompt_start[:, None].to(device)
    output_prompt_end = output_prompt_end[:, None].to(device)

    prompt_mask = torch.where(
        (output_ids_indices_per_row >= output_prompt_start) & (output_ids_indices_per_row < output_prompt_end),
        1, 0
    ).to(device)
    
    completion_mask = torch.where(
        output_ids_indices_per_row >= output_prompt_end,
        1,0
    ).to(device)

    return argparse.Namespace(attention_mask=attention_mask.int(),
        prompt_mask=prompt_mask.int(),
        completion_mask=completion_mask.int())

######################################################################################
# DO NOT IMPLEMENT THESE! Instead, implement output_ids_to_masks() instead.
# Correct implementations for them share some code, and they all have the same test
# cases. It might be easier to debug each component output separately though, so
# separate functions are provided.
######################################################################################
def output_ids_to_prompt_mask(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    """Returns the prompt mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        **kwargs)).get("prompt_mask", None)
def output_ids_to_completion_mask(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    """Returns the completion mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        **kwargs)).get("completion_mask", None)
def output_ids_to_attention_mask(*, output_ids, prompt_input_ids, pad_token_id, **kwargs):
    """Returns the attention mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        **kwargs)).get("attention_mask", None)
######################################################################################
######################################################################################
######################################################################################

def get_per_token_logprobs(*, logits, token_ids, **kwargs):
    """Returns a BSx(T-1) tensor of log-probabilities, whose ij element is the
    log-probability of the j+1th token in the ith sequence of [token_ids], given all
    the preceeding tokens 0...j (inclusive) in the ith sequence of [token_ids].

    HINTS:
    1. Understanding logits: singlestore.com/blog/a-guide-to-softmax-activation-function
    2. torch.gather() is your friend: pytorch.org/docs/stable/generated/torch.gather.html
    3. Prefer log-softmax() over log(softmax()) for numerical stability

    Args:
    logits      -- BSxTxV tensor of logits output by an LM with vocab size V given
                    [token_ids] as input
    token_ids   -- BSxT tensor of token IDs
    **kwargs    -- unused
    """
    #raise NotImplementedError("Implement me!")
    # convert scores to probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # predicting next token doesnt need last logit or first token ID
    logits_pred = log_probs[:, :-1, :]
    next_tokens = token_ids[:, 1:]

    # get probability for token
    final_probs = torch.gather(logits_pred, -1, next_tokens.unsqueeze(-1))
    return final_probs.squeeze(-1)

def grpo_rl_objective(*, logprobs_model, logprobs_pi_old, advantages, grpo_clip=0.2, **kwargs):
    """Returns a BSxL tensor where the ij element is the GRPO policy objective.

    HINT: This is (equation 3) in the original GRPO paper (arxiv.org/pdf/2402.03300),
        but with two changes (1) the KL term is not computed here, and (2) it is
        defined over batches log-probabilities and advantages that don't necessarily
        correspond to a single group (ie. those from one prompt) of model outputs.

    Args:
    Suppose that originally there was a BSx(L+1) tensor of tokens fed to LMs pi_theta
    and pi_old. The ith sequences in [logprobs_model] and [logprobs_pi_old] contain
    conditional log-probabilities for the (zero-indexed) tokens 1...L in the ith
    sequence of this original tensor. Moreover, the ith element of [advantages]
    contains the advantage computed for completion contained within the ith sequence
    of this original tensor.

    logprobs_model  -- BSxL tensor of log-probabilities from the model pi_theta
    logprobs_pi_old -- BSxL tensor of log-probabilities from the old policy pi_old
    advantages      -- BS-length tensor of advantages
    grpo_clip       -- clipping parameter for GRPO
    **kwargs        -- unused
    """
    #raise NotImplementedError("Implement me!")
    # calculate the probability ratio
    ratio = torch.exp(logprobs_model - logprobs_pi_old)

    # get advantages
    if advantages.shape != ratio.shape:
        adv = advantages.unsqueeze(1)
    else:
        adv = advantages
    
    # paper: (Pi / Pi_old) * A_hat
    part1 = ratio * adv

    # paper: clip(Pi / Pi_old, 1-e, 1+e) * A_hat
    ratio_c = torch.clamp(ratio, 1 - grpo_clip, 1 + grpo_clip)
    part2 = ratio_c * adv

    return torch.min(part1, part2)

def gpro_kl_loss(*, logprobs_model, logprobs_pi_ref, **kwargs):
    """Returns a BSxT tensor where the ij element is the modified KL loss for jth
    index of the ith element in the batch. Hint: equation (4) in the GRPO paper
    (arxiv.org/pdf/2402.03300.pdf).

    Args:
    Suppose that originally there was a BSx(T+1) tensor of tokens fed to LMs pi_theta
    and pi_ref. The ith sequences in [logprobs_model] and [logprobs_pi_ref] contain
    conditional log-probabilities for the (zero-indexed) tokens 1...T in the ith
    sequence of this original tensor.

    logprobs_model  -- BSxT tensor of log-probabilities from the model pi_theta
    logprobs_pi_ref -- BSxT tensor of log-probabilities from the model pi_ref
    **kwargs        -- unused
    """
    #raise NotImplementedError("Implement me!")
    # formula: exp(log_ref - log_model) - (log_ref - log_model) - 1
    log_ratio = logprobs_pi_ref - logprobs_model
    ratio = torch.exp(log_ratio)

    return ratio - log_ratio - 1

def rewards_to_advantages(*, rewards, grpo_completions, **kwargs):
    """Returns a (BS C)-length tensor of advantages computed from [rewards] using
    GRPO's method.

    HINT: Add 1e-4 to the standard deviation for numerical stability.

    Args:
    rewards             --(BS C)-length tensor of rewards
    grpo_completions    -- number of completions per prompt
    """
    #raise NotImplementedError("Implement me!")
    # split into sections of size grpo_completions
    matrix = rewards.view(-1, grpo_completions)

    # caclulate mean and std
    mean = matrix.mean(dim=1, keepdim=True)
    std = matrix.std(dim=1, keepdim=True)
    advantages = (matrix - mean) / (std + 1e-4)
    return advantages.view(-1)

@torch.no_grad()
def test_fn(*, fn, fn_args, fn_kwargs, fn_expected_output, device="cpu", verbose=False):
    """Tests [fn] with the given arguments and returns the output."""
    try:
        output = fn(*fn_args, device=device, **fn_kwargs)
    except Exception as e:
        tb = traceback.TracebackException.from_exception(e)
        error_trace = "".join(tb.format())
        tqdm.write("Function raised exception during evaluation of test inputs. Traceback:")
        tqdm.write(error_trace)
        return argparse.Namespace(
            passed=False,
            expected_output=fn_expected_output,
            actual_output=None)

    try:
        passed = torch.allclose(output, fn_expected_output, rtol=1e-4, atol=1e-6)
    except Exception as e:
        passed = False
        tb = traceback.TracebackException.from_exception(e)
        error_trace = "".join(tb.format())
        tqdm.write("Function output could not be compared to expected output. Traceback:")
        tqdm.write(error_trace)
    return argparse.Namespace(
        passed=passed,
        expected_output=fn_expected_output,
        actual_output=output)

from Utils import twrite
if __name__ == "__main__":
    possible_functions = ["get_per_token_logprobs",
        "grpo_rl_objective", "gpro_kl_loss",
        "rewards_to_advantages",

        # These are really testing just the output_ids_to_masks() function, but
        # it may be easier to work on its components separately.
        "output_ids_to_attention_mask",
        "output_ids_to_completion_mask",
        "output_ids_to_prompt_mask"]
    P = argparse.ArgumentParser()
    P.add_argument("--fns_to_test", choices=possible_functions, nargs="+",
        default=possible_functions,
        help="List of functions to test.")
    P.add_argument("--device", type=str, default="cpu",
        help="Device to use for testing.")

    # Two options: (1) test_case_data.pt: uses torch.load() to get the test case data.
    # I only 99.999% trust this to work on arbitrary systems. (2) test_case_data.json:
    # custom (de)serialization functions in UtilsBase.py to reconstruct tensor data.
    P.add_argument("--test_case_data_path", type=str,
        default=osp.join(osp.dirname(__file__), "test_case_data_fixed_Dec4-1813.pt"),
        help="Path to the test case data")
    P.add_argument("--seed", type=int, default=42,
        help="Random seed for testing case generation.")
    P.add_argument("-v", "--verbose", action="store_true",
        help="Whether to print verbose output during testing.")
    P.add_argument("-vv", "--verbose2", action="store_true",
        help="Whether to print extra verbose output during testing.")
    args = P.parse_args()
    args.verbose = args.verbose or args.verbose2

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    if args.test_case_data_path.endswith(".pt"):
        tqdm.write(f"Loading test case data from PT file {args.test_case_data_path}...")
        fname2test_data = torch.load(args.test_case_data_path)
    elif args.test_case_data_path.endswith(".json"):
        tqdm.write(f"Loading test case data from JSON file {args.test_case_data_path}...")
        with open(args.test_case_data_path, "r+") as f:
            fname2test_data = json.load(f)
            fname2test_data = UtilsBase.from_json_encoding(fname2test_data)
    else:
        raise ValueError(f"Unknown test_case_data_path extension for {args.test_case_data_path}")

    failed_tests, total_tests = 0, 0
    for fn_name in args.fns_to_test:
        test_cases = fname2test_data[fn_name]
        fn = globals()[fn_name]

        print(f"Function={fn_name}: Running {len(test_cases)} test cases...")
        for idx,test_case in enumerate(test_cases):
            total_tests += 1
            fn_args = test_case["fn_args"]
            fn_kwargs = test_case["fn_kwargs"]
            fn_expected_output = test_case["fn_expected_output"]

            fn_result = test_fn(fn=fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_expected_output=fn_expected_output, device=device, verbose=args.verbose)
            failed_tests += int(not fn_result.passed)
            test_result_str = "PASSED" if fn_result.passed else "FAILED"
            
            if not args.verbose and not args.verbose2:
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
            elif args.verbose and not args.verbose2:
                tqdm.write(f"\t=======================================================")
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
                tqdm.write(f"\tExpected output:\n{fn_result.expected_output}")
                tqdm.write(f"\tGot output:\n{fn_result.actual_output}")
            elif args.verbose2:
                tqdm.write(f"\t=======================================================")
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
                tqdm.write(f"\tFunction positional arguments (total={len(fn_args)}):")
                for arg_idx,a in enumerate(fn_args):
                    print_newline = "\n" if isinstance(a, torch.Tensor) and a.ndim > 1 else ""
                    tqdm.write(f"Arg {arg_idx}: {print_newline}{a}")
                tqdm.write(f"\tFunction keyword arguments (total={len(fn_kwargs)}):")
                for k,v in fn_kwargs.items():
                    print_newline = "\n" if isinstance(v, torch.Tensor) and v.ndim > 1 else ""
                    tqdm.write(f"{k}={print_newline}{v}")
                tqdm.write(f"\t=== Expected output:\n{fn_result.expected_output}")
                tqdm.write(f"\t=== Received output:\n{fn_result.actual_output}")

    tqdm.write(f"Testing complete: {total_tests-failed_tests}/{total_tests} tests passed.")
