import torch
from torch.utils.tensorboard import SummaryWriter
from modeling.cnn import BowelSoundCNN
from modeling.lstm import BowelSoundLSTM

import matplotlib.pyplot as plt

# Hyperparameters
num_classes = 4
batch_size = 2
mel_bands = 40
time_steps = 25

from torchview import draw_graph
from modeling.cnn import BowelSoundCNN

cnn_model = BowelSoundCNN(num_classes=num_classes)

# Torchview directly builds a graph
graph = draw_graph(
    cnn_model, input_size=(2, 1, mel_bands, time_steps), expand_nested=True  # batch=2
)

graph.visual_graph.render("cnn_model_architecture", format="png")
print("✅ Saved as cnn_model_architecture.png")


# # TensorBoard writer
# writer = SummaryWriter(log_dir="runs/bowel_sound_models")

# # CNN Model
# cnn_model = BowelSoundCNN(num_classes=num_classes).eval()
# dummy_cnn_input = torch.randn(batch_size, 1, mel_bands, time_steps)  # [B, C, H, W]

# # Add CNN graph
# writer.add_graph(cnn_model, dummy_cnn_input)
# print("CNN graph added to TensorBoard.")

# # LSTM Model
# lstm_model = BowelSoundLSTM(num_classes=num_classes, input_size=mel_bands).eval()
# dummy_lstm_input = torch.randn(
#     batch_size, time_steps, mel_bands
# )  # [B, seq_len, input_size]

# # Use torch.jit.script to handle control flow in forward (not output is generated otherwise)
# scripted_lstm = torch.jit.script(lstm_model)

# # Add LSTM graph
# writer.add_graph(scripted_lstm, dummy_lstm_input)
# print("LSTM graph added to TensorBoard.")


# # Finish
# writer.flush()
# writer.close()
# print(
#     "1) CMD in main project folder.\n2) Run `tensorboard --logdir=runs`\n3) and check the 'Graphs' tab."
# )
