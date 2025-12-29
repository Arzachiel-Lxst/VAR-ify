"""
VAR Analysis System
Detects HANDBALL and OFFSIDE violations in football videos
With MULTIPROCESSING for faster analysis
"""
import cv2
import json
import shutil
import numpy as np
import subprocess
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import os


def convert_to_h264(input_path: str, output_path: str):
    """Convert video to H.264 for browser compatibility"""
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac', output_path
        ], capture_output=True, check=True)
        return Path(output_path).exists()
    except Exception as e:
        print(f"[FFmpeg] Conversion failed: {e}")
        # Fallback: just copy the file
        try:
            shutil.copy(input_path, output_path)
            return Path(output_path).exists()
        except:
            return False


def extract_clip_with_text(video_path: str, output_path: str, start_sec: float, duration: float, text: str):
    """Extract clip from video with text overlay using ffmpeg"""
    try:
        # Use ffmpeg to extract clip with text overlay
        subprocess.run([
            'ffmpeg', '-y',
            '-ss', str(max(0, start_sec - 0.5)),
            '-i', video_path,
            '-t', str(duration + 1),
            '-vf', f"drawtext=text='{text}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10",
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac',
            output_path
        ], capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False


def create_simple_var_clip(video_path: str, output_path: str, events: list, event_type: str):
    """Create simple VAR clip by extracting from original video - NO text overlay"""
    if not events:
        return False
    
    # Get first event
    event = events[0]
    start_sec = max(0, event.timestamp - 1)
    
    try:
        # Simple clip extraction without text overlay (more compatible)
        result = subprocess.run([
            'ffmpeg', '-y',
            '-ss', str(start_sec),
            '-i', video_path,
            '-t', '4',  # 4 second clip
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"FFmpeg clip error: {e}")
        return False

from app.var.contact_detector import ContactVARAnalyzer
from app.var.offside_detector import OffsideVARAnalyzer


def _analyze_handball(video_path: str):
    """Worker function for handball detection"""
    analyzer = ContactVARAnalyzer()
    return analyzer.analyze(video_path, skip_frames=2)  # Skip 2 frames - ACCURATE


def _analyze_offside(video_path: str):
    """Worker function for offside detection"""
    analyzer = OffsideVARAnalyzer()
    return analyzer.analyze(video_path, skip_frames=3)  # Skip 3 frames - ACCURATE


class VARSystem:
    """Unified VAR Analysis System"""
    
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def cleanup_video_results(self, video_name: str):
        """Delete old results for specific video only (before re-processing)"""
        stem = Path(video_name).stem
        for file in self.output_dir.glob(f"{stem}*"):
            try:
                file.unlink()
            except:
                pass
    
    def analyze(self, video_path: str, create_video: bool = True) -> dict:
        """
        Full VAR analysis - detects both HANDBALL and OFFSIDE
        
        Args:
            video_path: Path to video file
            create_video: Whether to create output video
            
        Returns:
            dict with handball and offside events
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            # Try in uploads folder
            video_path = Path("data/uploads") / video_path.name
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Cleanup old results for THIS video only
        self.cleanup_video_results(video_path.name)
        
        print("\n" + "="*60)
        print("⚽ VAR ANALYSIS SYSTEM")
        print("="*60)
        print(f"📹 Video: {video_path.name}")
        print(f"⏱️  Started: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        # Run HANDBALL and OFFSIDE detection in PARALLEL using ThreadPoolExecutor
        print("\n[PARALLEL] 🚀 Running HANDBALL + OFFSIDE detection simultaneously...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            handball_future = executor.submit(_analyze_handball, str(video_path))
            offside_future = executor.submit(_analyze_offside, str(video_path))
            
            # Wait for both to complete
            handball_events = handball_future.result()
            offside_events = offside_future.result()
        
        print(f"      ✅ HANDBALL: {len(handball_events)} event(s)")
        print(f"      ✅ OFFSIDE: {len(offside_events)} event(s)")
        
        # Prepare result
        min_hand = float(os.getenv("MIN_HANDBALL_CONF", "0.6"))
        min_off = float(os.getenv("MIN_OFFSIDE_CONF", "0.6"))
        handball_events = [e for e in handball_events if getattr(e, "confidence", 0) >= min_hand]
        offside_events = [e for e in offside_events if getattr(e, "confidence", 0) >= min_off]
        result = {
            "video": video_path.name,
            "analyzed_at": datetime.now().isoformat(),
            "handball": [asdict(e) for e in handball_events],
            "offside": [asdict(e) for e in offside_events],
            "summary": {
                "total_handball": len(handball_events),
                "total_offside": len(offside_events),
                "total_violations": len(handball_events) + len(offside_events)
            }
        }
        
        # Save JSON
        json_path = self.output_dir / f"{video_path.stem}_VAR.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n📄 JSON saved: {json_path}")
        
        # Create video with intro + highlights (ALWAYS create video)
        if create_video:
            video_output = self.output_dir / f"{video_path.stem}_VAR.mp4"
            print(f"\n🎬 Creating VAR video...")
            
            # Create video with OpenCV directly to MP4
            self._create_var_video(str(video_path), handball_events, offside_events, str(video_output))
            
            # Ensure browser-compatible H.264 + AAC
            h264_path = self.output_dir / f"{video_path.stem}_VAR_h264.mp4"
            if convert_to_h264(str(video_output), str(h264_path)):
                try:
                    shutil.move(str(h264_path), str(video_output))
                except:
                    pass
            
            if video_output.exists():
                size = video_output.stat().st_size / 1024 / 1024
                print(f"✅ Video saved: {video_output} ({size:.1f} MB)")
            else:
                print("⚠️ Video creation failed")
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: dict):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("📊 VAR ANALYSIS SUMMARY")
        print("="*60)
        
        summary = result["summary"]
        
        if summary["total_handball"] > 0:
            print(f"\n🖐️  HANDBALL: {summary['total_handball']} detected")
            for i, h in enumerate(result["handball"][:5], 1):
                print(f"    #{i} | {h['timestamp']:.2f}s | {h['confidence']:.0%}")
        
        if summary["total_offside"] > 0:
            print(f"\n🚩 OFFSIDE: {summary['total_offside']} detected")
            for i, o in enumerate(result["offside"][:5], 1):
                print(f"    #{i} | {o['timestamp']:.2f}s | {o['confidence']:.0%}")
        
        if summary["total_violations"] == 0:
            print("\n✅ NO VIOLATIONS DETECTED")
        else:
            print(f"\n⚠️  TOTAL VIOLATIONS: {summary['total_violations']}")
        
        print("="*60)
    
    def _create_var_video(self, video_path: str, handball_events: list, offside_events: list, output_path: str):
        """Create combined VAR video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use H.264 codec for browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        RED = (0, 0, 255)
        YELLOW = (0, 255, 255)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (255, 100, 0)
        ORANGE = (0, 165, 255)
        
        # Intro (shorter - 1 second)
        for _ in range(int(fps * 1)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = [25, 15, 15]
            
            cv2.rectangle(frame, (width//2-80, 60), (width//2+80, 135), BLUE, -1)
            cv2.putText(frame, "VAR", (width//2-55, 115), cv2.FONT_HERSHEY_SIMPLEX, 2, WHITE, 4)
            cv2.putText(frame, "ANALYSIS", (width//2-85, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 2)
            
            cv2.putText(frame, f"HANDBALL: {len(handball_events)}", (width//2-90, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE, 2)
            cv2.putText(frame, f"OFFSIDE: {len(offside_events)}", (width//2-80, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            out.write(frame)
        
        # Process only top 2 events for speed
        for i, event in enumerate(handball_events[:2], 1):
            self._render_handball_event(cap, out, event, i, fps, width, height)
        
        for i, event in enumerate(offside_events[:2], 1):
            self._render_offside_event(cap, out, event, i, fps, width, height)
        
        # Verdict
        self._render_verdict(out, handball_events, offside_events, fps, width, height)
        
        cap.release()
        out.release()
    
    def _create_var_video_fast(self, video_path: str, handball_events: list, offside_events: list, output_path: str):
        """FAST video creation - minimal processing"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use H.264 codec for browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        ORANGE = (0, 165, 255)
        
        # Collect all event frames
        event_frames = {}
        for e in handball_events[:1]:  # Only 1 handball
            event_frames[e.frame] = ('HANDBALL', e)
        for e in offside_events[:1]:  # Only 1 offside
            event_frames[e.frame] = ('OFFSIDE', e)
        
        if not event_frames:
            cap.release()
            out.release()
            return
        
        # Just extract clips around events
        for frame_num in sorted(event_frames.keys()):
            event_type, event = event_frames[frame_num]
            start = max(0, frame_num - int(fps * 0.5))
            end = frame_num + int(fps * 0.5)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Simple overlay
                color = ORANGE if event_type == 'HANDBALL' else RED
                cv2.rectangle(frame, (10, 10), (200, 50), (0,0,0), -1)
                cv2.putText(frame, f"VAR: {event_type}", (15, 38), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                out.write(frame)
        
        cap.release()
        out.release()
    
    def _render_handball_event(self, cap, out, event, idx, fps, width, height):
        """Render handball event to video"""
        ORANGE = (0, 165, 255)
        YELLOW = (0, 255, 255)
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        BLACK = (0, 0, 0)
        
        # Quick intro (0.5s)
        for _ in range(int(fps * 0.5)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = [25, 25, 45]
            cv2.putText(frame, f"HANDBALL #{idx}", (width//2-110, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, ORANGE, 2)
            cv2.putText(frame, f"{event.timestamp:.2f}s | {event.confidence:.0%}", (width//2-100, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
            out.write(frame)
        
        # Shorter footage clip
        center = event.frame
        start = max(0, center - int(0.3 * fps))
        end = center + int(0.7 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        for idx_frame in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            
            if abs(idx_frame - center) < 8:
                bx, by = int(event.ball_pos[0]), int(event.ball_pos[1])
                hx, hy = int(event.hand_pos[0]), int(event.hand_pos[1])
                cv2.circle(frame, (bx, by), 15, YELLOW, 3)
                cv2.circle(frame, (hx, hy), 12, ORANGE, -1)
                cv2.line(frame, (bx, by), (hx, hy), RED, 2)
                cv2.putText(frame, "HANDBALL", (hx-50, hy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
            
            # HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 40), (45, 35, 35), -1)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
            cv2.rectangle(frame, (8, 6), (50, 34), ORANGE, -1)
            cv2.putText(frame, "VAR", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2)
            cv2.putText(frame, f"HANDBALL #{idx}", (60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            out.write(frame)
    
    def _render_offside_event(self, cap, out, event, idx, fps, width, height):
        """Render offside event to video"""
        RED = (0, 0, 255)
        YELLOW = (0, 255, 255)
        WHITE = (255, 255, 255)
        
        # Quick intro (0.5s)
        for _ in range(int(fps * 0.5)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = [45, 25, 25]
            cv2.putText(frame, f"OFFSIDE #{idx}", (width//2-100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 2)
            cv2.putText(frame, f"{event.timestamp:.2f}s | {event.confidence:.0%}", (width//2-100, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
            out.write(frame)
        
        # Shorter footage clip
        center = event.frame
        start = max(0, center - int(0.3 * fps))
        end = center + int(0.7 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        bbox = event.attacker_bbox if hasattr(event, 'attacker_bbox') and event.attacker_bbox else None
        
        for idx_frame in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            
            if abs(idx_frame - center) < 10:
                # Only draw box around offside player (no dots/circles)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 3)
                    cv2.putText(frame, "OFFSIDE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            
            # HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 40), (45, 35, 35), -1)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
            cv2.rectangle(frame, (8, 6), (50, 34), RED, -1)
            cv2.putText(frame, "VAR", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
            cv2.putText(frame, f"OFFSIDE #{idx}", (60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            out.write(frame)
    
    def _render_verdict(self, out, handball_events, offside_events, fps, width, height):
        """Render final verdict"""
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        ORANGE = (0, 165, 255)
        
        total = len(handball_events) + len(offside_events)
        
        for _ in range(int(fps * 1.5)):  # Shorter verdict
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                b = int(40 * (1 - y/height))
                frame[y, :] = [b, b//3, b//4]
            
            cv2.putText(frame, "VAR DECISION", (width//2-130, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3)
            
            y = 110
            if handball_events:
                cv2.rectangle(frame, (60, y), (width-60, y+45), ORANGE, 2)
                cv2.putText(frame, f"HANDBALL: {len(handball_events)}", (80, y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE, 2)
                y += 60
            
            if offside_events:
                cv2.rectangle(frame, (60, y), (width-60, y+45), RED, 2)
                cv2.putText(frame, f"OFFSIDE: {len(offside_events)}", (80, y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            
            if total > 0:
                cv2.rectangle(frame, (width//2-130, height-95), (width//2+130, height-50), RED, -1)
                cv2.putText(frame, f"{total} VIOLATION(S)", (width//2-95, height-63), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)
            else:
                cv2.rectangle(frame, (width//2-100, height-95), (width//2+100, height-50), GREEN, -1)
                cv2.putText(frame, "NO VIOLATIONS", (width//2-85, height-63), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
            
            out.write(frame)


def main():
    """Main entry point"""
    import sys
    
    print("\n" + "="*60)
    print("⚽ VAR ANALYSIS SYSTEM")
    print("="*60)
    
    # Find videos in uploads
    upload_dir = Path("data/uploads")
    videos = list(upload_dir.glob("*.mp4"))
    
    if not videos:
        print("❌ No videos found in data/uploads/")
        print("   Please add a video file to analyze.")
        return
    
    print(f"\n📁 Found {len(videos)} video(s) in data/uploads/:")
    for i, v in enumerate(videos, 1):
        size = v.stat().st_size / 1024 / 1024
        print(f"   {i}. {v.name} ({size:.1f} MB)")
    
    # Initialize VAR system
    var = VARSystem()
    
    # Process all videos
    for video in videos:
        var.analyze(str(video))
    
    print("\n✅ ALL DONE!")


if __name__ == "__main__":
    main()
