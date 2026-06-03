import os
import cv2
import numpy as np
from src.video_keyframe.video_decoder import VideoDecoder
from src.video_keyframe.scene_detector import HSVSceneDetector, TransitionDetector, compute_frame_quality
from src.video_keyframe.keyframe_selector import KeyFrameSelector


def generate_test_video_with_fades(output_path, duration_sec=10, fps=15):
    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    scenes = [
        ((255, 0, 0), "Scene 1: Red"),
        ((0, 255, 0), "Scene 2: Green"),
        ((0, 0, 255), "Scene 3: Blue"),
        ((255, 255, 0), "Scene 4: Yellow"),
    ]
    
    scene_duration = 2
    fade_duration = 0.5
    
    frames_per_scene = int(scene_duration * fps)
    frames_per_fade = int(fade_duration * fps)
    
    for scene_idx, (color, text) in enumerate(scenes):
        for i in range(frames_per_scene):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = color
            cv2.putText(frame, text, (width // 4, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        if scene_idx < len(scenes) - 1:
            next_color = scenes[scene_idx + 1][0]
            for i in range(frames_per_fade):
                alpha = i / frames_per_fade
                blended = tuple(int(c1 * (1 - alpha) + c2 * alpha) for c1, c2 in zip(color, next_color))
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = blended
                cv2.putText(frame, "FADING...", (width // 3, height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
    
    out.release()
    print(f"[OK] 测试视频已生成: {output_path}")
    return output_path


def test_old_vs_new(video_path):
    print("\n" + "=" * 70)
    print("[TEST] 对比测试: 旧算法 vs 新算法")
    print("=" * 70)
    
    print("\n[INFO] 场景信息:")
    print(f"   - 4个静态场景 (各2秒)")
    print(f"   - 3个淡入淡出转场 (各0.5秒)")
    print(f"   - 期望提取: 4个关键帧 (每个场景1个)")
    
    print("\n" + "-" * 70)
    print("[OLD] 旧算法 (无转场检测):")
    print("-" * 70)
    
    selector_old = KeyFrameSelector(
        video_path=video_path,
        output_dir="test_old_algo",
        detector_type="hsv",
        threshold=0.2,
        min_scene_length=0.3,
        detect_transitions=False,
        deduplicate=False
    )
    keyframes_old = selector_old.select_keyframes()
    stats_old = selector_old.get_stats()
    
    print(f"   提取的关键帧数量: {len(keyframes_old)}")
    print(f"   期望: 4, 实际: {len(keyframes_old)}")
    print(f"   多余: {max(0, len(keyframes_old) - 4)} 个模糊/重复帧")
    
    for i, kf in enumerate(keyframes_old[:10]):
        print(f"   #{i+1}. 帧 {kf.index:4d} @ {kf.timestamp:5.2f}s 清晰度={kf.sharpness:.0f}")
    
    print("\n" + "-" * 70)
    print("[NEW] 新算法 (带转场检测 + 去重):")
    print("-" * 70)
    
    selector_new = KeyFrameSelector(
        video_path=video_path,
        output_dir="test_new_algo",
        detector_type="hsv",
        threshold=0.2,
        min_scene_length=0.3,
        detect_transitions=True,
        transition_threshold=0.1,
        blur_threshold=80.0,
        deduplicate=True,
        duplicate_threshold=0.85
    )
    keyframes_new = selector_new.select_keyframes()
    stats_new = selector_new.get_stats()
    
    print(f"   提取的关键帧数量: {len(keyframes_new)}")
    print(f"   转场期间跳过的帧: {stats_new.get('skipped_transitions', 0)}")
    print(f"   过滤的模糊帧: {stats_new.get('skipped_blurry', 0)}")
    print(f"   过滤的重复帧: {stats_new.get('skipped_duplicates', 0)}")
    
    for i, kf in enumerate(keyframes_new):
        print(f"   #{i+1}. 帧 {kf.index:4d} @ {kf.timestamp:5.2f}s 清晰度={kf.sharpness:.0f}")
    
    print("\n" + "=" * 70)
    print("[RESULT] 改进效果:")
    print("=" * 70)
    
    old_extra = max(0, len(keyframes_old) - 4)
    new_extra = max(0, len(keyframes_new) - 4)
    print(f"   旧算法多提取了 {old_extra} 个不必要的帧")
    print(f"   新算法多提取了 {new_extra} 个不必要的帧")
    print(f"   减少了 {old_extra - new_extra} 个冗余帧")
    
    if len(keyframes_new) == 4:
        print("   [PASS] 新算法完美! 精确提取了4个关键帧!")
    elif len(keyframes_new) < len(keyframes_old):
        print(f"   [PASS] 新算法有改进!")
    else:
        print("   [WARN] 需要调整参数")
    
    print("\n[INFO] 生成的画廊:")
    print(f"   旧算法: test_old_algo/index.html")
    print(f"   新算法: test_new_algo/index.html")
    
    return len(keyframes_old), len(keyframes_new)


def analyze_transitions(video_path):
    print("\n" + "=" * 70)
    print("[ANALYZE] 转场检测分析")
    print("=" * 70)
    
    detector = HSVSceneDetector(threshold=0.3)
    transition_detector = TransitionDetector(window_size=5, transition_threshold=0.1)
    
    with VideoDecoder(video_path) as decoder:
        in_transition = False
        transition_count = 0
        
        print("\n帧变化分数序列 (转场期间预期位置):")
        for frame in decoder.iter_frames(sample_interval=1):
            change = detector.detect_change(frame)
            score = change.score if change else 0.0
            in_trans, mean_score = transition_detector.is_in_transition(score, frame.index)
            
            quality = compute_frame_quality(frame, blur_threshold=100)
            
            if in_trans and not in_transition:
                in_transition = True
                print(f"\n  [TRANS] 转场开始 @ 帧 {frame.index}")
            elif not in_trans and in_transition:
                in_transition = False
                transition_count += 1
                print(f"  [TRANS] 转场结束 @ 帧 {frame.index}")
                print(f"  [STATS] 平均变化分数: {mean_score:.3f}")
            
            if in_trans:
                status = "[TRANS]"
            elif score > 0.1:
                status = "[CHANGE]"
            else:
                status = "   "
            
            if frame.index % 5 == 0:
                print(f"{status} 帧 {frame.index:3d}: {score:.3f} 清晰度={quality.sharpness_score:.0f}")
        
        print(f"\n[STATS] 检测到 {transition_count} 个转场")
    
    return transition_count


def main():
    test_video = "test_fades.mp4"
    
    if os.path.exists(test_video):
        os.remove(test_video)
    
    generate_test_video_with_fades(test_video, duration_sec=10, fps=15)
    
    analyze_transitions(test_video)
    
    test_old_vs_new(test_video)
    
    print("\n[DONE] 测试完成!")


if __name__ == "__main__":
    main()
