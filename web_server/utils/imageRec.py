import os
import time
import glob
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import shutil

# Load ONNX model
def loadModel():
    """
    Load ONNX model from directory
    """
    session = ort.InferenceSession("./trained_models/v8_white_bg.onnx")
    return session

# Draw bounding boxes and labels
def drawOwnBox(img, x1, y1, x2, y2, label, colour=(36, 255, 12), text_colour=(0, 0, 0)):
    """
    Draws a bounding box with a label on the image.
    """
    name_to_id = {
        "NA": 'NA', "Bullseye": 99, "One": 11, "Two": 12, "Three": 13,
        "Four": 14, "Five": 15, "Six": 16, "Seven": 17, "Eight": 18, "Nine": 19,
        "A": 20, "B": 21, "C": 22, "D": 23, "E": 24, "F": 25, "G": 26, "H": 27,
        "S": 28, "T": 29, "U": 30, "V": 31, "W": 32, "X": 33, "Y": 34, "Z": 35,
        "Up": 36, "Down": 37, "Right": 38, "Left": 39, "Stop": 40
    }
    label = label + "-" + str(name_to_id.get(label, 'NA'))
    x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
    rand = str(int(time.time()))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"image_results/raw_image_{label}_{rand}.jpg", img)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), colour, -1)
    img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colour, 1)
    cv2.imwrite(f"image_results/annotated_image_{label}_{rand}.jpg", img)


def apply_canny(image, threshold1=30, threshold2=100):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_color

# Predict using ONNX model
def predictImage(image, session):
    """
    Predicts objects in an image using ONNX model.
    """
    img = Image.open(os.path.join('uploads', image))
    edges = apply_canny(img)
    fused_img = cv2.addWeighted(np.array(img), 0.8, edges, 0.2, 0)
    img_array = np.array(fused_img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    df_results = outputs[0]
    df_results = sorted(df_results, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)
    pred = 'NA'
    if df_results:
        for row in df_results:
            if row[4] > 0.5:
                pred = row
                break
        if isinstance(pred, (list, np.ndarray)):
            drawOwnBox(np.array(img), pred[0], pred[1], pred[2], pred[3], str(pred[5]))
    name_to_id = {
        "NA": 'NA', "Bullseye": 99, "One": 11, "Two": 12, "Three": 13,
        "Four": 14, "Five": 15, "Six": 16, "Seven": 17, "Eight": 18, "Nine": 19,
        "A": 20, "B": 21, "C": 22, "D": 23, "E": 24, "F": 25, "G": 26, "H": 27,
        "S": 28, "T": 29, "U": 30, "V": 31, "W": 32, "X": 33, "Y": 34, "Z": 35,
        "Up": 36, "Down": 37, "Right": 38, "Left": 39, "Stop": 40
    }
    return str(name_to_id.get(str(pred[5]), 'NA')) if isinstance(pred, (list, np.ndarray)) else 'NA'

# Stitch detected images
def stitchImage():
    """
    Stitches detected images together.
    """
    imgFolder = 'runs'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')
    imgPaths = glob.glob(os.path.join(imgFolder, "detect/*/*.jpg"))
    images = [Image.open(x) for x in imgPaths]
    total_width = sum(i.width for i in images)
    max_height = max(i.height for i in images)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.width
    stitchedImg.save(stitchedPath)
    for img in imgPaths:
        shutil.move(img, os.path.join("runs", "originals", os.path.basename(img)))
    return stitchedImg

#KL
# import os
# import time
# import glob
# import onnxruntime as ort
# import numpy as np
# from PIL import Image
# import cv2
# import shutil

# def loadModel():
#     """
#     Load ONNX model from directory
#     """
#     model_path = "./trained_models/model.onnx"
#     session = ort.InferenceSession(model_path)
#     return session

# def drawOwnBox(img, x1, y1, x2, y2, label, colour=(36,255,12), text_colour=(0,0,0)):
#     """
#     Draw the bounding box on the image and add the text label, saving both the raw and annotated image in the "image_results" folder
    
#     Inputs
#     ------
#     img: numpy.ndarray - image on which the bounding box is to be drawn

#     x1: int - x coordinate of the top left corner of the bounding box

#     y1: int - y coordinate of the top left corner of the bounding box

#     x2: int - x coordinate of the bottom right corner of the bounding box

#     y2: int - y coordinate of the bottom right corner of the bounding box

#     label: str - label to be written on the bounding box

#     color: tuple - color of the bounding box

#     text_color: tuple - color of the text label

#     Returns
#     -------
#     None

#     """
#     name_to_id = {
#         "NA": 'NA',
#         "Bullseye": 10,
#         "One": 11,
#         "Two": 12,
#         "Three": 13,
#         "Four": 14,
#         "Five": 15,
#         "Six": 16,
#         "Seven": 17,
#         "Eight": 18,
#         "Nine": 19,
#         "A": 20,
#         "B": 21,
#         "C": 22,
#         "D": 23,
#         "E": 24,
#         "F": 25,
#         "G": 26,
#         "H": 27,
#         "S": 28,
#         "T": 29,
#         "U": 30,
#         "V": 31,
#         "W": 32,
#         "X": 33,
#         "Y": 34,
#         "Z": 35,
#         "Up": 36,
#         "Down": 37,
#         "Right": 38,
#         "Left": 39,
#         "Up Arrow": 36,
#         "Down Arrow": 37,
#         "Right Arrow": 38,
#         "Left Arrow": 39,
#         "Stop": 40
#     }
#     # Reformat the label to {label name}-{label id}
#     label = label + "-" + str(name_to_id[label])
#     # Convert the coordinates to int
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     # Create a random string to be used as the suffix for the image name, just in case the same name is accidentally used
#     rand = str(int(time.time()))

#     # Save the raw image
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f"image_results/raw_image_{label}_{rand}.jpg", img)

#     # Draw the bounding box
#     img = cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
#     # For the text background, find space required by the text so that we can put a background with that amount of width.
#     (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#     # Print the text  
#     img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), colour, -1)
#     img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colour, 1)
#     # Save the annotated image
#     cv2.imwrite(f"image_results/annotated_image_{label}_{rand}.jpg", img)

# def preprocess_image(image):
#     """
#     Preprocess the image for ONNX model input
#     """
#     img = Image.open(os.path.join('uploads', image))
#     img = img.resize((640, 640))  # Resize to the input size of the model
#     img = np.array(img).astype('float32')
#     img = img / 255.0  # Normalize to [0, 1]
#     img = np.transpose(img, (2, 0, 1))  # Change to (C, H, W) format
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# def predictImage(image, model):
#     """
#     image: filename for the image 
#     model: loaded ONNX model
#     """
#     img = preprocess_image(image)
#     input_name = model.get_inputs()[0].name
#     outputs = model.run(None, {input_name: img})
    
#     # Assuming the model outputs bounding boxes and labels
#     boxes, labels, scores = outputs[0], outputs[1], outputs[2]
    
#     pred = 'NA'
#     for box, label, score in zip(boxes, labels, scores):
#         if score > 0.5:
#             x1, y1, x2, y2 = box
#             drawOwnBox(np.array(Image.open(os.path.join('uploads', image))), x1, y1, x2, y2, label)
#             pred = label
#             break

#     name_to_id = {
#         "NA": 'NA',
#         "Bullseye": 10,
#         "Right": 38,
#         "Left": 39,
#         "Right Arrow": 38,
#         "Left Arrow": 39,
#     }
#     image_id = str(name_to_id.get(pred, 'NA'))
#     return image_id

# def stitchImage():
#     """
#     Stitches the images in the folder together and saves it into runs/stitched folder
#     """
#     # Initialize path to save stitched image
#     imgFolder = 'runs'
#     stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

#     # Find all files that ends with ".jpg" (this won't match the stitched images as we name them ".jpeg")
#     imgPaths = glob.glob(os.path.join(imgFolder+"/detect/*/", "*.jpg"))
#     # Open all images
#     images = [Image.open(x) for x in imgPaths]
#     # Get the width and height of each image
#     width, height = zip(*(i.size for i in images))
#     # Calculate the total width and max height of the stitched image, as we are stitching horizontally
#     total_width = sum(width)
#     max_height = max(height)
#     stitchedImg = Image.new('RGB', (total_width, max_height))
#     x_offset = 0

#     # Stitch the images together
#     for im in images:
#         stitchedImg.paste(im, (x_offset, 0))
#         x_offset += im.size[0]
#     # Save the stitched image to the path
#     stitchedImg.save(stitchedPath)

#     # Move original images to "originals" subdirectory
#     for img in imgPaths:
#         shutil.move(img, os.path.join(
#             "runs", "originals", os.path.basename(img)))
#     return stitchedImg


# import os
# import time
# import glob
# import torch
# from utils.consts import MODEL_NAME
# import numpy as np
# from PIL import Image
# import cv2
# import shutil

# def loadModel():
#     """"
#     Load model from directory
#     """
#     model = torch.hub.load("./trained_models", 'custom', MODEL_NAME)
#     return model

# def drawOwnBox(img, x1, y1, x2, y2, label, colour=(36,255,12), text_colour=(0,0,0)):
#     """
#     Draw the bounding box on the image and add the text label, saving both the raw and annotated image in the "image_results" folder
    
#     Inputs
#     ------
#     img: numpy.ndarray - image on which the bounding box is to be drawn

#     x1: int - x coordinate of the top left corner of the bounding box

#     y1: int - y coordinate of the top left corner of the bounding box

#     x2: int - x coordinate of the bottom right corner of the bounding box

#     y2: int - y coordinate of the bottom right corner of the bounding box

#     label: str - label to be written on the bounding box

#     color: tuple - color of the bounding box

#     text_color: tuple - color of the text label

#     Returns
#     -------
#     None

#     """
#     name_to_id = {
#         "NA": 'NA',
#         "Bullseye": 10,
#         "One": 11,
#         "Two": 12,
#         "Three": 13,
#         "Four": 14,
#         "Five": 15,
#         "Six": 16,
#         "Seven": 17,
#         "Eight": 18,
#         "Nine": 19,
#         "A": 20,
#         "B": 21,
#         "C": 22,
#         "D": 23,
#         "E": 24,
#         "F": 25,
#         "G": 26,
#         "H": 27,
#         "S": 28,
#         "T": 29,
#         "U": 30,
#         "V": 31,
#         "W": 32,
#         "X": 33,
#         "Y": 34,
#         "Z": 35,
#         "Up": 36,
#         "Down": 37,
#         "Right": 38,
#         "Left": 39,
#         "Up Arrow": 36,
#         "Down Arrow": 37,
#         "Right Arrow": 38,
#         "Left Arrow": 39,
#         "Stop": 40
#     }
#     # Reformat the label to {label name}-{label id}
#     label = label + "-" + str(name_to_id[label])
#     # Convert the coordinates to int
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     # Create a random string to be used as the suffix for the image name, just in case the same name is accidentally used
#     rand = str(int(time.time()))

#     # Save the raw image
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f"image_results/raw_image_{label}_{rand}.jpg", img)

#     # Draw the bounding box
#     img = cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
#     # For the text background, find space required by the text so that we can put a background with that amount of width.
#     (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#     # Print the text  
#     img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), colour, -1)
#     img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colour, 1)
#     # Save the annotated image
#     cv2.imwrite(f"image_results/annotated_image_{label}_{rand}.jpg", img)


# def predictImage(image, model):
#     """
#     image: filename for the image 
#     model: loaded model
#     """
#     img = Image.open(os.path.join('uploads', image))
#     results = model(img)
#     results.save("runs")

#     df_results = results.pandas().xyxy[0]
#     df_results['bboxHt'] = df_results['ymax'] - df_results['ymin']
#     df_results['bboxWt'] = df_results['xmax'] - df_results['xmin']
#     df_results['bboxArea'] = df_results['bboxHt'] * df_results['bboxWt']
#     df_results = df_results.sort_values('bboxArea', ascending=False)
#     pred_list = df_results 
#     pred = 'NA'

#     if pred_list.size != 0:
#         # Go through the predictions, and choose the first one with confidence > 0.5
#         for _, row in pred_list.iterrows():
#             if row['name'] != 'Bullseye' and row['confidence'] > 0.5:
#                 pred = row    
#                 break

#         # Draw the bounding box on the image 
#         if not isinstance(pred,str):
#             drawOwnBox(np.array(img), pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['name'])
        
#     # Dictionary is shorter as only two symbols, left and right are needed
#     name_to_id = {
#         "NA": 'NA',
#         "Bullseye": 10,
#         "Right": 38,
#         "Left": 39,
#         "Right Arrow": 38,
#         "Left Arrow": 39,
#     }
#     # Return the image id
#     if not isinstance(pred,str):
#         image_id = str(name_to_id[pred['name']])
#     else:
#         image_id = 'NA'
#     return image_id

# def stitchImage():
#     """
#     Stitches the images in the folder together and saves it into runs/stitched folder
#     """
#     # Initialize path to save stitched image
#     imgFolder = 'runs'
#     stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

#     # Find all files that ends with ".jpg" (this won't match the stitched images as we name them ".jpeg")
#     imgPaths = glob.glob(os.path.join(imgFolder+"/detect/*/", "*.jpg"))
#     # Open all images
#     images = [Image.open(x) for x in imgPaths]
#     # Get the width and height of each image
#     width, height = zip(*(i.size for i in images))
#     # Calculate the total width and max height of the stitched image, as we are stitching horizontally
#     total_width = sum(width)
#     max_height = max(height)
#     stitchedImg = Image.new('RGB', (total_width, max_height))
#     x_offset = 0

#     # Stitch the images together
#     for im in images:
#         stitchedImg.paste(im, (x_offset, 0))
#         x_offset += im.size[0]
#     # Save the stitched image to the path
#     stitchedImg.save(stitchedPath)

#     # Move original images to "originals" subdirectory
#     for img in imgPaths:
#         shutil.move(img, os.path.join(
#             "runs", "originals", os.path.basename(img)))
#     return stitchedImg
