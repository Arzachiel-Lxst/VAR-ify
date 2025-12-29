"""
VAR-ify ML Service - Hugging Face Spaces
Video Assistant Referee Analysis with YOLOv8 and MediaPipe
"""

import os
import tempfile
import gradio as gr
from pathlib import Path

# Import VAR analyzers
from app.var.handball_detector import HandballVARAnalyzer
from app.var.offside_detector import OffsideVARAnalyzer
from run_var import VARAnalyzer

# Initialize analyzer
analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = VARAnalyzer(output_dir="results")
    return analyzer


def analyze_video(video_file, analysis_type="both"):
    """
    Analyze video for VAR violations
    
    Args:
        video_file: Uploaded video file
        analysis_type: "handball", "offside", or "both"
    
    Returns:
        results dict and output video path
    """
    if video_file is None:
        return {"error": "No video uploaded"}, None
    
    try:
        var = get_analyzer()
        
        # Run analysis
        results = var.analyze(
            video_path=video_file,
            check_handball=(analysis_type in ["handball", "both"]),
            check_offside=(analysis_type in ["offside", "both"]),
            create_video=True
        )
        
        # Format results
        output = {
            "status": "completed",
            "handball_events": len(results.get("handball_events", [])),
            "offside_events": len(results.get("offside_events", [])),
            "violations_found": results.get("handball_events", []) + results.get("offside_events", []),
            "video_url": results.get("video_url")
        }
        
        video_path = results.get("video_url")
        if video_path and os.path.exists(video_path):
            return output, video_path
        
        return output, None
        
    except Exception as e:
        return {"error": str(e)}, None


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="VAR-ify", theme=gr.themes.Dark()) as demo:
        gr.Markdown("""
        # ⚽ VAR-ify - Video Assistant Referee
        
        Upload a football/soccer video to detect **handball** and **offside** violations using AI.
        
        **Features:**
        - 🤚 Handball detection using MediaPipe pose estimation
        - 🏃 Offside detection using YOLOv8 player tracking
        - 📹 Generates highlight video with detected events
        
        **Limitations:**
        - Max video duration: 15 seconds
        - Supported formats: MP4, AVI, MOV
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                analysis_type = gr.Radio(
                    choices=["both", "handball", "offside"],
                    value="both",
                    label="Analysis Type"
                )
                analyze_btn = gr.Button("🔍 Analyze Video", variant="primary")
            
            with gr.Column():
                results_output = gr.JSON(label="Analysis Results")
                video_output = gr.Video(label="VAR Highlight Video")
        
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, analysis_type],
            outputs=[results_output, video_output]
        )
        
        gr.Markdown("""
        ---
        **Note:** Processing may take 30-60 seconds depending on video length.
        
        Built with YOLOv8, MediaPipe, and Gradio.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
