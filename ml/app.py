"""
VAR-ify ML Service - Hugging Face Spaces
Video Assistant Referee Analysis with YOLOv8 and MediaPipe
"""

import os
import tempfile
import gradio as gr
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

# Import VAR system
from run_var import VARSystem

# Initialize FastAPI for API endpoints
api_app = FastAPI()
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = VARSystem(output_dir="results")
    return analyzer


@api_app.post("/api/analyze")
async def api_analyze(video: UploadFile = File(...)):
    """API endpoint for backend to call"""
    # Save uploaded video
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    video_path = upload_dir / video.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    try:
        var = get_analyzer()
        results = var.analyze(str(video_path), create_video=True)
        
        handball_events = results.get("handball", [])
        offside_events = results.get("offside", [])
        
        # Find output video
        video_stem = video_path.stem
        result_video = f"results/{video_stem}_VAR.mp4"
        
        return {
            "success": True,
            "handball_events": len(handball_events),
            "offside_events": len(offside_events),
            "handball_details": handball_events,
            "offside_details": offside_events,
            "video_url": result_video if os.path.exists(result_video) else None,
            "summary": results.get("summary", {})
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        # Cleanup uploaded file
        if video_path.exists():
            video_path.unlink()


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
        
        print(f"[DEBUG] Analyzing video: {video_file}")
        
        # Run analysis
        results = var.analyze(
            video_path=video_file,
            create_video=True
        )
        
        print(f"[DEBUG] Analysis results: {results}")
        
        # Format results based on analysis_type
        handball_events = results.get("handball", [])
        offside_events = results.get("offside", [])
        
        if analysis_type == "handball":
            offside_events = []
        elif analysis_type == "offside":
            handball_events = []
        
        # Format output
        output = {
            "status": "completed",
            "handball_events": len(handball_events),
            "offside_events": len(offside_events),
            "handball_details": handball_events,
            "offside_details": offside_events,
            "summary": results.get("summary", {})
        }
        
        # Find output video - check multiple possible locations
        video_stem = Path(video_file).stem
        possible_paths = [
            f"results/{video_stem}_VAR.mp4",
            f"data/results/{video_stem}_VAR.mp4",
            Path("results") / f"{video_stem}_VAR.mp4",
        ]
        
        video_path = None
        for p in possible_paths:
            if os.path.exists(str(p)):
                video_path = str(p)
                print(f"[DEBUG] Found video at: {video_path}")
                break
        
        if video_path is None:
            print(f"[DEBUG] No video found. Checked: {possible_paths}")
        
        return output, video_path
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[ERROR] Analysis failed: {error_msg}")
        return {"error": str(e), "traceback": error_msg}, None


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="VAR-ify", theme="dark") as demo:
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
    # Mount FastAPI app for API endpoints
    app = gr.mount_gradio_app(api_app, demo, path="/")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
