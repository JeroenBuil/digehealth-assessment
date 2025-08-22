import torch
from torch.utils.tensorboard import SummaryWriter
from modeling.cnn import BowelSoundCNN
from modeling.lstm import BowelSoundLSTM

# Hyperparameters
num_classes = 4
batch_size = 2
mel_bands = 40
time_steps = 25

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/bowel_sound_models")

# Visualize CNN
cnn_model = BowelSoundCNN(num_classes=num_classes)
dummy_cnn_input = torch.randn(batch_size, 1, mel_bands, time_steps)  # [B, C, H, W]
writer.add_graph(cnn_model, dummy_cnn_input)
print("CNN graph added to TensorBoard.")

# Visualize LSTM
lstm_model = BowelSoundLSTM(num_classes=num_classes, input_size=mel_bands)
dummy_lstm_input = dummy_cnn_input  # same shape [B, 1, H, W]


# The LSTM forward expects [B, seq_len, input_size], so define a lambda wrapper
def lstm_forward(x):
    return lstm_model(x)


writer.add_graph(lstm_model, dummy_lstm_input)
print("LSTM graph added to TensorBoard.")

# Close the writer
writer.close()
print("Done! Run `tensorboard --logdir=runs` to visualize.")
