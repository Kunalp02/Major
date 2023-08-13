import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Define a simple policy network for text summarization
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return self.softmax(x)

# Hyperparameters
learning_rate = 0.001
num_epochs = 50
max_summary_length = 50  # Maximum length of generated summary

# Load pre-trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Instantiate policy network
policy_net = PolicyNetwork(input_size=768, hidden_size=128, output_size=max_summary_length)

# Define optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Example input text
input_text = "Artificial intelligence (AI) is intelligence demonstrated by machines."

# Convert input text to embeddings using the pre-trained model
input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
with torch.no_grad():
    input_embeddings = model.get_input_embeddings()(input_ids)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Generate summary lengths using policy network
    summary_lengths_probs = policy_net(input_embeddings)
    summary_lengths_dist = torch.distributions.Categorical(summary_lengths_probs)
    summary_lengths = summary_lengths_dist.sample()
    
    # Generate summaries using T5 model
    summary_ids = model.generate(input_ids, max_length=summary_lengths[0], num_beams=4, early_stopping=True)
    
    # Calculate reward based on a simple metric (you might need a more sophisticated reward function)
    reward = 1.0 / len(summary_ids[0])
    
    # Compute loss using REINFORCE algorithm
    loss = -summary_lengths_dist.log_prob(summary_lengths) * reward
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Generate final summary using learned policy
sampled_length = summary_lengths_probs.argmax(dim=1)
final_summary_ids = model.generate(input_ids, max_length=sampled_length[0], num_beams=4, early_stopping=True)
final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
print("Final Generated Summary:", final_summary)
