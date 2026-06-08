import os
import shutil
import uuid
import threading
import time
import json
from PIL import Image
from dotenv import load_dotenv

from flask import send_file
from pyngrok import ngrok
from flask import Flask, request, jsonify

load_dotenv()

from inference import (
    get_model_paths,
    initialize_face_detector,
    initialize_mvdiffusion_pipeline,
    initialize_gslrm_model,
    setup_camera_parameters,
    process_single_image,
)
import torch

print("starting...")

# ngrok authentication
ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])

# Disconnect all active tunnels before starting, then kill the process.
# ngrok.kill() alone only stops the local binary but leaves remote endpoints
# alive on ngrok's servers, causing ERR_NGROK_334 on the next run.
try:
    for tunnel in ngrok.get_tunnels():
        ngrok.disconnect(tunnel.public_url)
except Exception:
    pass
ngrok.kill()
time.sleep(1)
app = Flask(__name__)
FLASK_PORT = 5001

STATUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_statuses")
os.makedirs(STATUS_DIR, exist_ok=True)


def _write_status(job_id, data):
    with open(os.path.join(STATUS_DIR, job_id + ".json"), "w") as f:
        json.dump(data, f)


def _read_status(job_id):
    path = os.path.join(STATUS_DIR, job_id + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# --- Load models once at startup ---
print("Initializing models (one-time startup)...")
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_mvdiffusion_path, _gslrm_ckpt_path, _gslrm_cfg_path = get_model_paths()
_face_detector = initialize_face_detector(_device)
_diffusion_pipeline, _generator, _color_prompt_embeddings = initialize_mvdiffusion_pipeline(
    _mvdiffusion_path, _device
)
_gslrm_model = initialize_gslrm_model(_gslrm_ckpt_path, _gslrm_cfg_path, _device)
_camera_intrinsics, _camera_extrinsics = setup_camera_parameters(_device)
_generator.manual_seed(4)
print("Models ready.")
# -----------------------------------


@app.route('/')
def home():
    return 'Flask app is running!'


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided in the request.'}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file in the request.'}), 400

    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_filename_no_ext = os.path.splitext(uploaded_file.filename)[0]

    input_dir = os.path.join(current_dir, "examples")
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_image_name_png = f"{original_filename_no_ext}.png"
    input_image_path = os.path.join(input_dir, input_image_name_png)

    try:
        temp_uploaded_path = os.path.join(input_dir, uploaded_file.filename)
        uploaded_file.save(temp_uploaded_path)

        img = Image.open(temp_uploaded_path)
        img.save(input_image_path)

        if temp_uploaded_path != input_image_path:
            os.remove(temp_uploaded_path)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to save image: {str(e)}'}), 500

    job_id = str(uuid.uuid4())
    _write_status(job_id, {'status': 'processing'})
    post_received_at = time.time()

    # Rename the saved image to job_id so concurrent jobs never share a path.
    job_image_name = f"{job_id}.png"
    job_image_path = os.path.join(input_dir, job_image_name)
    os.rename(input_image_path, job_image_path)

    def run_pipeline():
        thread_start = time.time()
        print(f"[TIMING] Job {job_id[:8]}: POST→thread start lag: {thread_start - post_received_at:.2f}s")
        try:
            process_single_image(
                image_file=job_image_name,
                input_dir=input_dir,
                output_dir=output_dir,
                auto_crop=True,
                unclip_pipeline=_diffusion_pipeline,
                generator=_generator,
                color_prompt_embedding=_color_prompt_embeddings,
                gs_lrm_model=_gslrm_model,
                demo_fxfycxcy=_camera_intrinsics,
                demo_c2w=_camera_extrinsics,
                guidance_scale_2D=3.0,
                step_2D=70,
                face_detector=_face_detector,
            )

            if os.path.exists(job_image_path):
                os.remove(job_image_path)

            ply_path = os.path.join(output_dir, job_id, "gaussians.ply")
            if not os.path.exists(ply_path):
                _write_status(job_id, {
                    'status': 'error',
                    'message': 'Pipeline completed but gaussians.ply not found.',
                })
                return

            t_write = time.time()
            _write_status(job_id, {
                'status': 'success',
                'ply_path': ply_path,
                'ply_filename': f"{original_filename_no_ext}_gaussians.ply",
            })
            print(f"[TIMING] Job {job_id[:8]}: status write: {time.time() - t_write:.3f}s")
            print(f"[TIMING] Job {job_id[:8]}: total wall time (POST→done): {time.time() - post_received_at:.2f}s")

        except Exception as e:
            if os.path.exists(job_image_path):
                os.remove(job_image_path)
            _write_status(job_id, {
                'status': 'error',
                'message': f'Unexpected error: {str(e)}',
            })

    threading.Thread(target=run_pipeline).start()
    return jsonify({'status': 'processing', 'job_id': job_id})


@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    result = _read_status(job_id)
    if result is None:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(result)


@app.route('/download/<job_id>', methods=['GET'])
@app.route('/download/<job_id>/ply', methods=['GET'])
def download(job_id):
    result = _read_status(job_id)
    if result is None or result['status'] != 'success':
        return jsonify({'status': 'not_found'}), 404
    return send_file(result['ply_path'], as_attachment=True, download_name=result['ply_filename'])


def run_flask():
    app.run(port=FLASK_PORT, debug=False, use_reloader=False)


# Start server in background thread
server_thread = threading.Thread(target=run_flask)
server_thread.daemon = True
server_thread.start()

time.sleep(2)

public_url = ngrok.connect(FLASK_PORT, pooling_enabled=True)
print(f"ngrok tunnel URL: {public_url}")
print("POST image to /process_image, then poll /status/<job_id> for results.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    ngrok.kill()
