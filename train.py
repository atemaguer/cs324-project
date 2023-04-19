import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, set_seed
from utils import get_model_tokenizer, load_training_dataset, DEFAULT_SEED

device = "cuda" if torch.cuda.is_available else "cpu"
tokenizer = None

class ContrastiveLossFinetuner(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        negatives = inputs["negative"].to(device)
        positives = inputs["positive"].to(device)
        anchors = inputs["anchor"].to(device)
        neg_mask = inputs["negative_mask"].to(device)
        pos_mask = inputs["positive_mask"].to(device)
        anch_mask = inputs["anchor_mask"].to(device)

        # forward pass
        anchor_outs =  F.normalize(model(anchors, attention_mask=anch_mask, output_hidden_states=True).last_hidden_state[:, -1, :], dim=1)
        negatives_outs = F.normalize(model(negatives, attention_mask=anch_mask, output_hidden_states=True).last_hidden_state[:, -1, :], dim=1)
        positives_outs = F.normalize(model(positives, attention_mask=anch_mask, output_hidden_states=True).last_hidden_state[:, -1, :], dim=1)

        #compute scores
        scores = (anchor_outs @ positives_outs.T) * torch.exp(torch.tensor(0.07))

        # compute custom loss (suppose one has 3 labels with different weights)
        labels = torch.arange(anchor_outs.size()[0], dtype=torch.long).to(device)

        loss = F.cross_entropy(scores, labels)

        outputs = {
            "logits": scores,
            "labels": labels
        }
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys
    ):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            if prediction_loss_only:
                return (loss, None, None)

            return (loss, None, outputs["labels"])

def train():

    global tokenizer

    set_seed(DEFAULT_SEED)

    model_checkpoint = "bigscience/bloomz-560m"
    local_output_dir = "bloomz-contrastive-finetuned"
    max_length = 256
    batch_size = 16

    model, tokenizer = get_model_tokenizer(model_checkpoint, device, base_model=True)
    split_dataset = load_training_dataset(tokenizer, max_length).train_test_split(test_size = 0.1, seed=DEFAULT_SEED)

    args = TrainingArguments(
        fp16=False,
        # bf16=True,
        output_dir=local_output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,

        logging_dir=f"{local_output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,

        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=1,
        remove_unused_columns=False,
        weight_decay=0.01,
        disable_tqdm=True,  # declutter the output a little

        # deepspeed="./deepspeed.json",

        report_to="tensorboard",
        push_to_hub=False,
        eval_accumulation_steps=1
        # local_rank=True
    )

    trainer = ContrastiveLossFinetuner(
                    model,
                    args,
                    train_dataset=split_dataset["train"],
                    eval_dataset=split_dataset["test"],
                )

    trainer.train()
    trainer.save_model(output_dir=local_output_dir)
    
def main():
    train()

if __name__ == "__main__":
    main()