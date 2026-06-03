import os
import sys
import click
from tqdm import tqdm
from .video_decoder import VideoDecoder
from .keyframe_selector import KeyFrameSelector
from .gallery_generator import GalleryGenerator


@click.group()
@click.version_option(version="0.1.0", prog_name="video-keyframe")
def main():
    """视频关键帧提取工具 - 基于场景检测的关键帧提取和画廊生成"""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_dir", default=None, help="输出目录")
@click.option("-t", "--threshold", type=float, default=0.3, help="场景检测阈值 (0-1)")
@click.option("-m", "--min-scene-length", type=float, default=1.0, help="最小场景时长(秒)")
@click.option("--max-keyframes", type=int, default=100, help="最大关键帧数量")
@click.option("--detector", type=click.Choice(["hsv", "combined"]), default="hsv", help="场景检测算法")
@click.option("--sample-interval", type=int, default=2, help="帧采样间隔")
@click.option("--gallery/--no-gallery", default=True, help="是否生成HTML画廊")
@click.option("--uniform", type=int, default=0, help="均匀采样帧数(0表示使用场景检测)")
@click.option("--detect-transitions/--no-detect-transitions", default=True, help="是否检测淡入淡出转场")
@click.option("--transition-threshold", type=float, default=0.12, help="转场检测阈值")
@click.option("--blur-threshold", type=float, default=50.0, help="模糊帧过滤阈值(拉普拉斯方差)")
@click.option("--deduplicate/--no-deduplicate", default=True, help="是否去除重复/相似帧")
@click.option("--duplicate-threshold", type=float, default=0.85, help="重复帧相似度阈值 (0-1)")
@click.option("--chapters/--no-chapters", default=False, help="是否生成智能章节(需要OCR)")
@click.option("--ocr-interval", type=int, default=30, help="OCR采样间隔(帧)")
@click.option("--min-chapter-duration", type=float, default=10.0, help="最小章节时长(秒)")
@click.option("--chapter-threshold", type=float, default=0.3, help="章节划分相似度阈值")
@click.option("--ocr-lang", type=str, default="ch_sim,en", help="OCR语言，逗号分隔")
@click.option("--gpu/--no-gpu", default=True, help="是否使用GPU加速OCR")
def extract(
    video_path,
    output_dir,
    threshold,
    min_scene_length,
    max_keyframes,
    detector,
    sample_interval,
    gallery,
    uniform,
    detect_transitions,
    transition_threshold,
    blur_threshold,
    deduplicate,
    duplicate_threshold,
    chapters,
    ocr_interval,
    min_chapter_duration,
    chapter_threshold,
    ocr_lang,
    gpu
):
    """从视频中提取关键帧"""
    output_dir = output_dir or os.path.splitext(os.path.basename(video_path))[0] + "_keyframes"
    
    click.echo(f"[VIDEO] 视频文件: {click.format_filename(video_path)}")
    
    with VideoDecoder(video_path) as decoder:
        info = decoder.get_info()
        click.echo(f"   分辨率: {info['width']}x{info['height']}")
        click.echo(f"   帧率: {info['fps']:.1f} FPS")
        click.echo(f"   时长: {info['duration']:.1f} 秒")
        click.echo(f"   总帧数: {info['frame_count']}")
    
    click.echo("")
    click.echo(f"[PROCESS] 提取关键帧...")
    if detect_transitions:
        click.echo(f"   [+] 转场检测: 开启 (阈值: {transition_threshold})")
    if blur_threshold > 0:
        click.echo(f"   [+] 模糊帧过滤: 开启 (阈值: {blur_threshold})")
    if deduplicate:
        click.echo(f"   [+] 重复帧去重: 开启 (阈值: {duplicate_threshold})")
    if chapters:
        click.echo(f"   [+] 智能章节: 开启 (语言: {ocr_lang})")
    
    ocr_languages = [lang.strip() for lang in ocr_lang.split(',')]
    
    selector = KeyFrameSelector(
        video_path=video_path,
        output_dir=output_dir,
        detector_type=detector,
        threshold=threshold,
        min_scene_length=min_scene_length,
        max_keyframes=max_keyframes,
        sample_interval=sample_interval,
        detect_transitions=detect_transitions,
        transition_threshold=transition_threshold,
        blur_threshold=blur_threshold,
        deduplicate=deduplicate,
        duplicate_threshold=duplicate_threshold,
        enable_chapters=chapters,
        ocr_languages=ocr_languages,
        ocr_interval=ocr_interval,
        min_chapter_duration=min_chapter_duration,
        chapter_similarity_threshold=chapter_threshold,
        use_gpu=gpu
    )
    
    pbar = tqdm(total=100, unit="%")
    last_progress = 0
    
    def progress_callback(progress, count, skipped=0):
        nonlocal last_progress
        delta = int(progress) - last_progress
        if delta > 0:
            pbar.update(delta)
            last_progress = int(progress)
            pbar.set_description(f"已找到 {count} 个关键帧")
    
    if uniform > 0:
        keyframes = selector.select_uniform(num_frames=uniform)
        pbar.update(100)
    else:
        keyframes = selector.select_keyframes(progress_callback=progress_callback)
    
    pbar.close()
    
    stats = selector.get_stats()
    
    click.echo("")
    click.echo(f"[RESULT] 提取完成! 共找到 {len(keyframes)} 个关键帧")
    
    if stats:
        if stats.get('skipped_transitions', 0) > 0:
            click.echo(f"   转场期间跳过: {stats['skipped_transitions']} 帧")
        if stats.get('skipped_blurry', 0) > 0:
            click.echo(f"   过滤模糊帧: {stats['skipped_blurry']} 帧")
        if stats.get('skipped_duplicates', 0) > 0:
            click.echo(f"   过滤重复帧: {stats['skipped_duplicates']} 帧")
        if stats.get('chapters_count', 0) > 0:
            click.echo(f"   生成章节: {stats['chapters_count']} 个")
    
    click.echo("")
    for i, kf in enumerate(keyframes[:10]):
        click.echo(f"   #{i+1:3d}. 帧 {kf.index:6d} @ {kf.timestamp:7.2f}s (清晰度: {kf.sharpness:.0f})")
    
    if len(keyframes) > 10:
        click.echo(f"   ... 还有 {len(keyframes) - 10} 更多关键帧")
    
    if chapters:
        chapter_list = selector.get_chapters()
        if chapter_list:
            click.echo("")
            click.echo("[CHAPTERS] 章节索引:")
            for chap in chapter_list:
                duration = chap.end_time - chap.start_time
                click.echo(f"   {chap.title}")
                click.echo(f"      时间: {chap.start_time:.2f}s - {chap.end_time:.2f}s (时长: {duration:.1f}s)")
                if chap.keywords:
                    click.echo(f"      关键词: {', '.join(chap.keywords[:5])}")
                click.echo(f"      关键帧: {len(chap.keyframes)} 个")
    
    if gallery and len(keyframes) > 0:
        click.echo("")
        click.echo(f"[GALLERY] 生成HTML画廊...")
        generator = GalleryGenerator(output_dir)
        video_name = os.path.basename(video_path)
        chapter_list = selector.get_chapters() if chapters else None
        html_path = generator.generate(
            keyframes, 
            selector.get_video_info(), 
            video_name,
            chapters=chapter_list
        )
        click.echo(f"[DONE] 画廊已生成: {click.format_filename(html_path)}")
    
    click.echo("")
    click.echo(f"[OUTPUT] 输出目录: {click.format_filename(output_dir)}")


@main.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
def info(video_path):
    """显示视频信息"""
    try:
        with VideoDecoder(video_path) as decoder:
            info = decoder.get_info()
        
        click.echo("[INFO] 视频信息:")
        click.echo(f"   文件: {click.format_filename(video_path)}")
        click.echo(f"   分辨率: {info['width']} x {info['height']}")
        click.echo(f"   帧率: {info['fps']:.2f} FPS")
        click.echo(f"   总帧数: {info['frame_count']}")
        click.echo(f"   时长: {info['duration']:.2f} 秒")
        hours = int(info['duration'] // 3600)
        minutes = int((info['duration'] % 3600) // 60)
        secs = int(info['duration'] % 60)
        click.echo(f"   格式化时长: {hours:02d}:{minutes:02d}:{secs:02d}")
    except Exception as e:
        click.echo(f"[ERROR] 错误: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("keyframes_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
def gallery(keyframes_dir, video_path):
    """从已提取的关键帧生成HTML画廊"""
    from .keyframe_selector import KeyFrame
    
    click.echo("[GALLERY] 生成HTML画廊...")
    
    frames_dir = os.path.join(keyframes_dir, "frames")
    thumbs_dir = os.path.join(keyframes_dir, "thumbnails")
    
    if not os.path.exists(frames_dir):
        click.echo(f"[ERROR] 找不到关键帧目录: {frames_dir}")
        sys.exit(1)
    
    keyframes = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    for fname in frame_files:
        frame_path = os.path.join(frames_dir, fname)
        thumb_path = os.path.join(thumbs_dir, fname) if os.path.exists(thumbs_dir) else ""
        
        parts = fname.replace('.jpg', '').split('_')
        try:
            idx = int(parts[-1]) if len(parts) > 1 else 0
        except ValueError:
            idx = 0
        
        keyframes.append(KeyFrame(
            index=idx,
            timestamp=0.0,
            score=1.0,
            image_path=frame_path,
            thumbnail_path=thumb_path
        ))
    
    with VideoDecoder(video_path) as decoder:
        video_info = decoder.get_info()
    
    generator = GalleryGenerator(keyframes_dir)
    video_name = os.path.basename(video_path)
    html_path = generator.generate(keyframes, video_info, video_name)
    
    click.echo(f"[DONE] 画廊已生成: {click.format_filename(html_path)}")


if __name__ == "__main__":
    main()
