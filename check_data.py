import json
from pathlib import Path

print("=" * 60)
print("DATA PREPARATION OUTPUT CHECK")
print("=" * 60)

# Check jersey crops
jersey_train = Path("data/jersey_crops/train_annotations.json")
jersey_val = Path("data/jersey_crops/val_annotations.json")

if jersey_train.exists():
    tj = json.load(open(jersey_train))
    vj = json.load(open(jersey_val))
    print("\n1. JERSEY CROPS:")
    print(f"   Train: {len(tj)} samples")
    print(f"   Val: {len(vj)} samples")
    
    if len(tj) > 0:
        print(f"\n   Example train sample:")
        print(f"     Image: {tj[0]['image_path']}")
        print(f"     Team: {tj[0]['team_label']}")
        print(f"     Track ID: {tj[0]['track_id']}")
        
        # Count teams
        team0 = sum(1 for x in tj if x['team_label'] == 0)
        team1 = sum(1 for x in tj if x['team_label'] == 1)
        print(f"\n   Team distribution (train):")
        print(f"     Team 0: {team0} samples")
        print(f"     Team 1: {team1} samples")
else:
    print("\n1. JERSEY CROPS: NOT FOUND")

# Check ball trajectories
ball_train = Path("data/ball_trajectories/train_trajectories.json")
ball_val = Path("data/ball_trajectories/val_trajectories.json")

if ball_train.exists():
    tb = json.load(open(ball_train))
    vb = json.load(open(ball_val))
    print("\n2. BALL TRAJECTORIES:")
    print(f"   Train: {len(tb)} trajectories")
    print(f"   Val: {len(vb)} trajectories")
    
    if len(tb) > 0:
        print(f"\n   Example trajectory:")
        print(f"     Sequence: {tb[0]['sequence_name']}")
        print(f"     Frames: {len(tb[0]['positions'])}")
        print(f"     First position: {tb[0]['positions'][0]}")
else:
    print("\n2. BALL TRAJECTORIES: NOT FOUND")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nData đã được prepare và lưu vào:")
print("  - data/jersey_crops/")
print("  - data/ball_trajectories/")
print("\nĐây là dữ liệu training, KHÔNG phải output video.")
print("Output video sẽ được tạo khi chạy main.py inference.")
