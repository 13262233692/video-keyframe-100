import os
from src.video_keyframe.keyframe_selector import KeyFrameSelector

selector = KeyFrameSelector(
    video_path='test_fades.mp4',
    output_dir='test_new_algo',
    detector_type='hsv',
    threshold=0.2,
    min_scene_length=0.3,
    detect_transitions=True,
    transition_threshold=0.1,
    blur_threshold=30.0,
    deduplicate=False,
    duplicate_threshold=0.75
)
keyframes = selector.select_keyframes()
stats = selector.get_stats()

print('New algorithm results:')
print('  Keyframes:', len(keyframes))
print('  Skipped transitions:', stats.get('skipped_transitions', 0))
print('  Skipped blurry:', stats.get('skipped_blurry', 0))
print('  Skipped duplicates:', stats.get('skipped_duplicates', 0))

for i, kf in enumerate(keyframes):
    print('  #%d. Frame %d @ %.2fs sharpness=%.0f' % (i+1, kf.index, kf.timestamp, kf.sharpness))
