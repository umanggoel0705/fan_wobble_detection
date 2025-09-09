from flask import Flask, request, render_template, Response, redirect, url_for, session
import final_script_tracker

app = Flask(__name__)
video_path_global = None

@app.route("/", methods=["GET", "POST"])
def home():
    global video_path_global

    if request.method == "POST":
        file = request.files.get("video")
        if not file:
            return render_template("home.html", output="No video uploaded")

        video_path = "uploaded_path.mp4"
        file.save(video_path)
        video_path_global = video_path

        # Run tracking and generate plots
        final_script_tracker.main(video_path)

        # Render the same home page but indicate that plots are ready
        print("FPS being passed to template:", final_script_tracker.FPS)
        return render_template("home.html", output="Tracking..", plots_ready=True, fps=final_script_tracker.FPS)

    return render_template("home.html")

@app.route("/results")
def results():
    if not video_path_global:
        return redirect(url_for("home"))
    
    # print("Analysis data being passed to template:", final_script.analysis_data)
    return render_template("results.html", analysis=final_script_tracker.analysis_data)

@app.route("/video_feed")
def video_feed():
    if not video_path_global:
        return "No video uploaded yet.", 400
    return Response(final_script_tracker.tracker(video_path_global, final_script_tracker.ROI),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed_roi")
def video_feed_roi():
    if not video_path_global:
        return "No video uploaded yet.", 400
    
    print("ROI is being passed to template:", final_script_tracker.ROI)
    return Response(final_script_tracker.tracker_roi(video_path_global, final_script_tracker.ROI),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
