import utils
import torch
import fsspec
import json

class Prompts():
    def __init__(self, prompts, tokenizer, config, device=None):
        self.device = device
   
        # find a token id that maps to a single char and then back
        # to a token
        self.mask_str = tokenizer.decode(
           torch.tensor([188]), add_special_tokens=False)[0]
        self.mask_token_id = tokenizer.encode(self.mask_str, add_special_tokens=False)[0]

        prompts = [p.strip() for p in prompts]
        prompts = self._pad_to_batch(prompts, config.loader.eval_batch_size)
        self._tokenize(tokenizer, config.model.length, prompts)

    def _pad_to_batch(self, prompts, batch_size):
        if len(prompts) < batch_size:
            pad = [""] * (batch_size - len(prompts))
            prompts += pad
        return prompts
    
    def _handle_trailing_prompt(self, prompt, num_tokens):
        if not prompt.startswith("<|endoftext|>") or prompt == "<|endoftext|>":
            return prompt
        p = prompt[len("<|endoftext|>"):]
        np = [self.mask_str] * (num_tokens - len(p))
        np.append(p)
        return "".join(np)

    def _rewrite_with_mask(self, prompt):
        idx = prompt.find("<|mask|>")
        if idx == -1:
            return prompt
        nidx = prompt.find(":")
        if nidx == -1:
            return prompt
        n = int(prompt[idx+8:nidx])
        mask = self.mask_str * n
        return prompt[:idx] + mask + prompt[nidx+1:]

    def _handle_masked_prompt(self, prompt):
        rewritten = ""
        while True:
            rewritten = self._rewrite_with_mask(prompt)
            if rewritten == prompt:
                break
            prompt = rewritten
        return rewritten
          
    def _tokenize(self, tokenizer, num_tokens, prompts):
        self.prompts = []
    
        for prompt in prompts:
            prompt = self._handle_trailing_prompt(prompt, num_tokens)
            prompt = self._handle_masked_prompt(prompt)
            self.prompts.append(prompt)

        tokenized = tokenizer.batch_encode_plus(self.prompts,
                                            padding='max_length',
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=num_tokens,
                                            return_tensors='pt')
        self.tokenized = tokenized.input_ids
        self.mask = tokenized.attention_mask.bool()
        self.lens = tokenized.attention_mask.sum(dim=1)
        input_mask = self.tokenized == self.mask_token_id
        self.mask = torch.logical_and(torch.logical_not(input_mask), self.mask)
        
        self.tokenized = self.tokenized.to(self.device)
        self.mask = self.mask.to(self.device)

        for i, p in enumerate(self.prompts):
            print(f"prompt:      {p}")
            print(f"mask:        {self.mask[i][:50]}")
            print(f"tokenized:   {self.tokenized[i][:50]}")
     
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, tokenizer, config, device=None):
        self.prompt_state = Prompts(prompts, tokenizer, config, device=device)
        self.mask_str = self.prompt_state.mask_str
        self.prompts = self.prompt_state.prompts
        self.tokenized = self.prompt_state.tokenized
        self.mask = self.prompt_state.mask
        self.lens = self.prompt_state.lens

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return (self.tokenized[idx], self.mask[idx], self.lens[idx])
    
    def decode_for_prompts(self, tokens, lens, tokenizer):
      out = []
      for t, l in zip(tokens, lens):
          pr = t[:l]
          v = tokenizer.decode(pr, skip_special_tokens=True)
          out.append(v)
      return out
    
    def decode_for_sentiment(self, tokens, lens, tokenizer):
      out = []
      for t, l in zip(tokens, lens):
          pr = t[:l]
          v = tokenizer.decode(pr, skip_special_tokens=True)
          out.append(v)
      return out


def prompt(model, config, logger, tokenizer):
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []

  with open(config.sampling.prompts_path, 'r') as f:
    prompts = f.readlines()

  ds = PromptDataset(prompts, tokenizer, config, device=model.device)
  dl = torch.utils.data.DataLoader(ds, batch_size=config.loader.eval_batch_size, shuffle=False)

  for (tokenized, masks, lens) in dl:
      def projection_fn(x):   
        y = torch.where(masks, tokenized, x)
        return y
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps, projection_fn=projection_fn)

      text_samples = ds.decode_for_prompts(samples, lens, tokenizer)
      model.metrics.record_entropy(samples)
      model.metrics.record_generative_perplexity(
        text_samples, config.model.length, model.device)

      all_samples.extend(list(text_samples))

  generative_ppl = 0.
  entropy = 0.
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
               'generated_seqs': all_samples}, f, indent=4)
  print('Samples saved at:', samples_path)