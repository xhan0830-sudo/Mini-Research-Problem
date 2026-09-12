import argparse
import os
import time
import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_benchmark(model_path, base_model_id="meta-llama/Meta-Llama-3-8B"):
  print("[INFO] Starting Evaluation Pipeline...")

  tokenizer = AutoTokenizer.from_pretrained(
      model_path if os.path.exists(model_path) else base_model_id
  )

  base_model = AutoModelForCausalLM.from_pretrained(
      base_model_id, torch_dtype=torch.float16, device_map="auto"
  )

  if os.path.exists(model_path):
    model = PeftModel.from_pretrained(base_model, model_path)
    print(f"[INFO] Successfully loaded LoRA weights from {model_path}")
  else:
    model = base_model
    print("[WARN] Model path not found. Evaluating baseline base model.")

  model.eval()

  # Test Evaluation Queries
  test_prompts = [
      "Classify system status: Voltage spike detected in power manifold.",
      "Classify system status: Thermal readings operating within nominal range.",
      "Classify system status: Critical pressure loss in primary hydraulic line.",
  ]
  ground_truth = [1]  # 1 = Anomaly/Risk, 0 = Normal

  predictions = []
  latencies = []

  with torch.no_grad():
    for prompt in test_prompts:
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

      start_time = time.time()
      outputs = model.generate(**inputs, max_new_tokens=15)
      latency = (time.time() - start_time) * 1000  # ms
      latencies.append(latency)

      decoded_text = tokenizer.decode(outputs, skip_special_tokens=True)
      pred = (
          1
          if any(
              k in decoded_text.lower()
              for k in ["spike", "critical", "loss", "risk"]
          )
          else 0
      )
      predictions.append(pred)

  acc = accuracy_score(ground_truth, predictions)
  f1 = f1_score(ground_truth, predictions, average="weighted")
  avg_latency = np.mean(latencies)

  print("\n" + "=" * 50)
  print("         BENCHMARK EVALUATION RESULTS        ")
  print("=" * 50)
  print(f" Accuracy:            {acc * 100:.2f}%")
  print(f" Weighted F1 Score:   {f1:.4f}")
  print(f" Avg Inference Speed: {avg_latency:.2f} ms/sample")
  if torch.cuda.is_available():
    print(
        f" Peak VRAM Allocated:  {torch.cuda.max_memory_allocated() / 1e9:.2f}"
        " GB"
    )
  print("=" * 50)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, default="./lora_output")
  args = parser.parse_args()
  evaluate_benchmark(args.model_path)
