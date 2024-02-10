import torch
from torch.utils.data import random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling
from torch.utils.tensorboard import SummaryWriter
import transformers
print(transformers.__version__)

# Step 1: Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('d:/huggingface_models/gpt2-hindi')


# Step 2: Tokenize the training data
tokenizer = GPT2Tokenizer.from_pretrained('d:/huggingface_models/gpt2-hindi')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Step 3: Prepare the training data
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="d:/test/chat/clean_chat_short.csv",
    block_size=128
)
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
import subprocess

# Specify the directory where your TensorBoard logs are located
#model_output_dir = '.gpt2hindi/results/tensorboard_logs'

# Launch TensorBoard using subprocess
# tensorboard_command = f'tensorboard --logdir {model_output_dir}/runs --host=localhost --port=6006'
# process = subprocess.Popen(tensorboard_command, shell=True)

# # Wait for TensorBoard to finish
# try:
#     process.wait()
# except KeyboardInterrupt:
#     process.terminate()


# Step 4: Create a TrainingArguments object
training_args = TrainingArguments(
    output_dir='./gpt2hindi/results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./gpt2hindi/results/logs',
    logging_steps=1000,
    save_steps=5000,
    evaluation_strategy='steps',
    eval_steps=5000,
    load_best_model_at_end=True,
    use_mps_device=torch.backends.mps.is_available(),
    report_to='tensorboard'
)
writer = SummaryWriter(log_dir='.gpt2hindi/result/tensorboard_logs')

# Create a SummaryWriter for TensorBoard logging

# Custom callback to log metrics to TensorBoard
# Custom callback to log metrics to TensorBoard
class CustomCallback:
    def __init__(self, writer):
        self.writer = writer

    def log_metrics_to_tensorboard(self, trainer):
        metrics = trainer.log_metrics("train", metrics=trainer.train_metrics)
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, trainer.step)

# Step 5: Instantiate a Trainer object with the custom callback
custom_callback = CustomCallback(writer)
from transformers import TensorBoardCallback
tensorboard_callback=TensorBoardCallback()
# Step 5: Instantiate a Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=tensorboard_callback,
)

# Step 6: Train the model
trainer.train()
model.save_pretrained("d:/mymodels/hindi/gpt2_fine_tune")
tokenizer.save_pretrained("d:/mymodels/hindi/gpt2_fine_tune")
#trainer.save_model("d:/mymodels/gpt2_fine_tune")


