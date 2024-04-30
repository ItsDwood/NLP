from torch import cuda, manual_seed
from util import get_dataloader, compute_metrics
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer

def main():

    # initialize model and dataloaders
    device = "cuda" if cuda.is_available() else "cpu"

    # seed the model before initializing weights so that your code is deterministic
    manual_seed(457)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_data = get_dataloader("train")
    dev_data = get_dataloader("dev")

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()