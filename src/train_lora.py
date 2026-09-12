import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
  parser = argparse.ArgumentParser(
      description="LoRA Fine-Tuning Pipeline for Causal Decoders"
  )
  parser.add_argument(
      "--base_model", type=str, default="meta-llama/Meta-Llama-3-8B"
  )
  parser.add_argument("--output_dir", type=str, default="./lora_output")
  parser.add_argument("--num_epochs", type=int, default=3)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--lr", type=float, default=2e-4)
  args = parser.parse_args()

  print(f"[INFO] Initializing tokenizer and base model: {args.base_model}")
  tokenizer = AutoTokenizer.from_pretrained(
      args.base_model, trust_remote_code=True
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
      args.base_model,
      torch_dtype=torch.bfloat16
      if torch.cuda.is_bf16_supported()
      else torch.float16,
      device_map="auto",
  )

  # Inject Low-Rank Adaptation (LoRA) adapter matrices
  peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      r=16,  # Rank dimension
      lora_alpha=32,  # Scaling factor
      lora_dropout=0.05,
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
  )

  model = get_peft_model(model, peft_config)
  print("[INFO] Trainable Parameter Summary:")
  model.print_trainable_parameters()

  # Load dataset sample
  raw_dataset = load_dataset("imdb", split="train[:1000]")

  def tokenize_fn(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )

  tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)

  training_args = TrainingArguments(
      output_dir=args.output_dir,
      num_train_epochs=args.num_epochs,
      per_device_train_batch_size=args.batch_size,
      gradient_accumulation_steps=4,
      learning_rate=args.lr,
      weight_decay=0.01,
      logging_steps=10,
      save_strategy="epoch",
      bf16=torch.cuda.is_bf16_supported(),
      fp16=not torch.cuda.is_bf16_supported(),
      report_to="none",
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset,
      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
  )

  print("[INFO] Starting LoRA Fine-Tuning Execution...")
  trainer.train()

  # Save adapter checkpoint
  model.save_pretrained(args.output_dir)
  tokenizer.save_pretrained(args.output_dir)
  print(f"[SUCCESS] LoRA adapter successfully saved to {args.output_dir}")


if __name__ == "__main__":
  main()
