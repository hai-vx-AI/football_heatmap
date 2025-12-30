import pandas as pd

df = pd.read_csv('output/logs/snmot117_1000frames_frames_summary.csv')

print('📊 FRAME STATISTICS (750 frames):')
print(f'\nAverage players: {df["num_players"].mean():.1f}')
print(f'Average referees: {df["num_referees"].mean():.2f}')
print(f'Average goalkeepers: {df["num_goalkeepers"].mean():.2f}')

ball_detected = (df['ball_status'] == 'detected').sum()
print(f'\n⚽ Ball detected: {ball_detected}/750 frames ({ball_detected*100/750:.1f}%)')

print('\nBall detection improvement:')
print(f'  Previous (400 frames): 185/400 = 46.2%')
print(f'  Current  (750 frames): {ball_detected}/750 = {ball_detected*100/750:.1f}%')
print(f'  Improvement: {ball_detected*100/750 - 46.2:+.1f}%')
