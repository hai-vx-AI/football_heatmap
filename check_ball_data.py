import json

# Check train data
with open('data/ball_trajectories/train_sequences.json', 'r') as f:
    train_data = json.load(f)

print("="*80)
print("TRAIN DATA")
print("="*80)
print(f"Total sequences: {train_data['metadata']['num_sequences']}")
print(f"Sequence length: {train_data['metadata']['seq_length']}")
print(f"Prediction horizon: {train_data['metadata']['pred_horizon']}")
print(f"Normalization mean: {train_data['normalization']['mean']}")
print(f"Normalization std: {train_data['normalization']['std']}")

# Check val data
with open('data/ball_trajectories/val_sequences.json', 'r') as f:
    val_data = json.load(f)

print("\n" + "="*80)
print("VAL DATA")
print("="*80)
print(f"Total sequences: {val_data['metadata']['num_sequences']}")
print(f"Sequence length: {val_data['metadata']['seq_length']}")
print(f"Prediction horizon: {val_data['metadata']['pred_horizon']}")
print(f"Normalization mean: {val_data['normalization']['mean']}")
print(f"Normalization std: {val_data['normalization']['std']}")

print("\n" + "="*80)
print("READY TO TRAIN!")
print("="*80)
