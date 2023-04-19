import torch
import logging
from functools import partial

from utils import load_evaluation_dataset, get_model_tokenizer, compute_metrics
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

device = "cuda" if torch.cuda.is_available else "cpu"

tokenizer = None

logger = logging.getLogger(__name__)

class RegularFinetuner(Seq2SeqTrainer):
  def compute_loss(self, model, inputs, return_outputs=False):
      labels = inputs["labels"].to(device)
      input_ids = inputs["input_ids"].to(device)
      attention_mask = inputs["attention_mask"].to(device)

      labels = torch.tensor([
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
      ]).to(device)

      # forward pass
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss

      return (loss, outputs) if return_outputs else loss

def evaluate():
    global tokenizer

    model_checkpoint = "./bloomz-contrastive-finetuned/bloomz-finetuned" #"bigscience/bloomz-560m" #
    max_length = 512
    batch_size = 8

    model, tokenizer = get_model_tokenizer(model_checkpoint, device)
    eval_dataset = load_evaluation_dataset(tokenizer, max_length).train_test_split(test_size = 0.02)

    local_output_dir = "bloomz-finetuned"

    args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,

        evaluation_strategy="steps",
        logging_dir=f"{local_output_dir}/logs",
        logging_strategy="steps",
        logging_steps=100,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,

        predict_with_generate=True,

        eval_accumulation_steps=1
    )

    trainer = RegularFinetuner(
        model,
        args,
        train_dataset=eval_dataset["train"],
        eval_dataset=eval_dataset["test"],
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer)
    )

    eval_results = trainer.evaluate()

    print(f"Evaluation results: {eval_results}")

def main():
    evaluate()

if __name__ == "__main__":
    main()

