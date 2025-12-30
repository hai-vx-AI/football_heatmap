import json
import pandas as pd

# Load frames data
frames = []
with open('output/logs/snmot117_400frames_frames.jsonl', 'r') as f:
    for line in f:
        frames.append(json.loads(line))

# Filter frames with ball detected
ball_frames = [f for f in frames if f['ball']['status'] == 'detected']

print(f"⚽ BALL DETECTION ANALYSIS")
print(f"{'='*60}")
print(f"\nTotal ball detections: {len(ball_frames)}/400 frames ({len(ball_frames)/4:.1f}%)")

# Calculate confidence stats
confidences = [f['ball']['conf'] for f in ball_frames]
print(f"\nConfidence statistics:")
print(f"  Min:  {min(confidences):.3f}")
print(f"  Max:  {max(confidences):.3f}")
print(f"  Mean: {sum(confidences)/len(confidences):.3f}")

# Show sample detections
print(f"\nSample ball detections (first 15 frames):")
print(f"{'Frame':<8} {'Confidence':<12} {'Position (x, y)':<20} {'Size (w x h)'}")
print(f"{'-'*70}")

for i, f in enumerate(ball_frames[:15]):
    frame_idx = f['frame_idx']
    conf = f['ball']['conf']
    bbox = f['ball']['bbox']
    center = f['ball']['center']
    
    if bbox:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        print(f"{frame_idx:<8} {conf:<12.3f} ({center[0]:.0f}, {center[1]:.0f}){'':<8} {width:.0f} x {height:.0f}")

# Analyze ball detection gaps
print(f"\n\nBall detection timeline (first 100 frames):")
timeline = ""
for i in range(min(100, len(frames))):
    if frames[i]['ball']['status'] == 'detected':
        timeline += "●"
    else:
        timeline += "○"
    if (i+1) % 50 == 0:
        timeline += "\n"

print(timeline)
print(f"\n● = Ball detected | ○ = Ball lost")

# Count consecutive detection/loss periods
detections_sequences = []
current_seq = 0
for f in frames:
    if f['ball']['status'] == 'detected':
        current_seq += 1
    else:
        if current_seq > 0:
            detections_sequences.append(current_seq)
            current_seq = 0
if current_seq > 0:
    detections_sequences.append(current_seq)

print(f"\nDetection sequences:")
print(f"  Number of sequences: {len(detections_sequences)}")
print(f"  Longest sequence: {max(detections_sequences) if detections_sequences else 0} frames")
print(f"  Average sequence: {sum(detections_sequences)/len(detections_sequences):.1f} frames" if detections_sequences else "  No sequences")
