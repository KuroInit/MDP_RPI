import io
import os
import sys
import cv2
import numpy as np
from flask import Flask, jsonify, request
try:
    from picamera import PiCamera  
    picamera_available = True
except ImportError:
    picamera_available = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'web_server', 'utils', 'trained_models', 'v8_white_bg.onnx')
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', 'static')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'result.jpg')  

ort_session = None
try:
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(MODEL_PATH)
    model_inputs = ort_session.get_inputs()
    input_name = model_inputs[0].name
except Exception as e:
    print(f"[ERROR] Loading Model Failed: {e}", file=sys.stderr)

CONF_THRESHOLD = 0.4

app = Flask(__name__)

@app.route('/capture', methods=['GET'])
def capture_and_detect():
    try:
        
        frame = None
        if picamera_available:
            camera = PiCamera()
            camera.resolution = (640, 480)  
            camera.capture(OUTPUT_IMAGE_PATH)
            camera.close()
            frame = cv2.imread(OUTPUT_IMAGE_PATH)
        else:
            cap = cv2.VideoCapture(0) 
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return jsonify({"error": "Camera Error: Cannot access camera"}), 500
            cv2.imwrite(OUTPUT_IMAGE_PATH, frame)

        if frame is None:
            return jsonify({"error": "Camera Error: Cannot access camera"}), 500

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          

        if ort_session is not None:
            input_shape = model_inputs[0].shape  
            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                target_height, target_width = input_shape[2], input_shape[3]
                img_resized = cv2.resize(img, (target_width, target_height))
            else:
                img_resized = img
        else:
            return jsonify({"error": "Model not loaded"}), 500

        img_data = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize
        # (H, W, C) -> (C, H, W)
        img_data = np.transpose(img_data, (2, 0, 1))
        # (C, H, W) -> (1, C, H, W)
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

        # Inference
        outputs = ort_session.run(None, {input_name: img_data})
       
        output_data = outputs[0]
        detections = []
        output_array = np.squeeze(output_data)
        if output_array.ndim == 1:
            output_array = np.expand_dims(output_array, axis=0)
        for det in output_array:
            if det.shape[-1] >= 6:
                x1, y1, x2, y2, score, class_id = det[:6]
            elif det.shape[-1] >= 5:
                x_center, y_center, width, height, score = det[:5]
                class_id = int(det[5]) if det.shape[-1] > 5 else 0
                # Convert to x1, y1, x2, y2
                x1 = x_center - width/2
                y1 = y_center - height/2
                x2 = x_center + width/2
                y2 = y_center + height/2
            else:
                continue
            
            if score < CONF_THRESHOLD:
                continue
            detections.append((float(x1), float(y1), float(x2), float(y2), float(score), int(class_id)))

        if not detections:
            
            return jsonify({"id": None, "image_path": OUTPUT_IMAGE_PATH, "message": "No Inference result"}), 200

        best_det = max(detections, key=lambda x: x[4])  
        x1, y1, x2, y2, best_score, best_class = best_det
        best_class_id = int(best_class)

        img_h, img_w = frame.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(img_w, int(x2)); y2 = min(img_h, int(y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.putText(frame, f"ID:{best_class_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imwrite(OUTPUT_IMAGE_PATH, frame)

        response = {
            "id": best_class_id,
            "image_path": OUTPUT_IMAGE_PATH
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Handle request error: {str(e)}")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)
