from roboflow import Roboflow

# Roboflow project: https://universe.roboflow.com/test-0leeq/mdp_cv
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("test-0leeq").project("mdp_cv")
version = project.version(2)
dataset = version.download("yolov8")