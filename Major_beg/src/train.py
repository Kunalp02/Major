import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader
from rpaper_dataset import ResearchPaperDataset  # Import your custom dataset class

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/bart-large-cnn"
num_epochs = 10
batch_size = 4
learning_rate = 1e-4

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

# Prepare your dataset
train_dataset = ResearchPaperDataset(
    data_path="data/dataset.csv",  # Specify your dataset path
    tokenizer=tokenizer,            # Pass the tokenizer
    max_length=512                  # Specify the maximum sequence length
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        targets = batch['target_ids'].to(device)
        loss = model(inputs, labels=targets).loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Save the trained model
model.save_pretrained("models")  # Save the model in the 'models' directory



# import tensorflow as tf
# from transformers import BartTokenizer, TFBartForConditionalGeneration, TFBartConfig
# from tensorflow.keras.utils import Sequence
# from rpaper_dataset import ResearchPaperDataset  # Import your custom dataset class

# # Configuration
# device = "cuda" if tf.config.experimental.list_physical_devices("GPU") else "cpu"
# model_name = "facebook/bart-large-cnn"
# num_epochs = 10
# batch_size = 4
# learning_rate = 1e-4

# # Load BART model and tokenizer
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = TFBartForConditionalGeneration.from_pretrained(model_name)

# # Prepare your dataset
# train_dataset = ResearchPaperDataset(
#     data_path="data/dataset.csv",  # Specify your dataset path
#     tokenizer=tokenizer,            # Pass the tokenizer
#     max_length=512                  # Specify the maximum sequence length
# )

# train_dataloader = tf.data.Dataset.from_generator(
#     lambda: train_dataset, output_signature=(
#         {
#             'input_ids': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#             'attention_mask': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#         },
#         tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#     )
# ).batch(batch_size).shuffle(buffer_size=len(train_dataset))

# # Optimizer and loss
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     total_loss = 0
#     for batch in train_dataloader:
#         with tf.device(device):
#             inputs = {
#                 'input_ids': batch['input_ids'],
#                 'attention_mask': batch['attention_mask']
#             }
#             targets = batch['target_ids']
#             with tf.GradientTape() as tape:
#                 logits = model(inputs, training=True).logits
#                 loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(targets, logits)
#             gradients = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#             total_loss += loss.numpy()
    
#     avg_loss = total_loss / len(train_dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# # Save the trained model
# model.save_pretrained("models")  # Save the model in the 'models' directory
