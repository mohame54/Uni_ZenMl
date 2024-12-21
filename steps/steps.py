import os
from zenml import step
import torch
from zenml.logger import get_logger
import mlflow
from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import mlflow
import mlflow.pytorch
import dagshub


app = Flask(__name__)
logger = get_logger(__name__)


def decode_base64_to_pil(base64_string:str)->Image:
    image_data = base64.b64decode(base64_string.encode("utf-8"))
    image_stream = BytesIO(image_data)
    pil_image = Image.open(image_stream)
    return pil_image


def process_image(image:np.array)->np.array:
    image = image/255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    return image


@step
def download_convert_model(local_dir: str = "models"):
    logger.info(
        f" logging into dagsHub"
    )
    mlflow.set_tracking_uri("https://dagshub.com/mohame54/ML_Exp_Uni.mlflow")
    dagshub.init(repo_owner='mohame54', repo_name='ML_Exp_Uni', mlflow=True)
    logger.info(
        f" Downloading the model to {local_dir}  dir..."
    )

    artifact_uri = 'runs:/6804bb7005e84a1e99aba5d6d138242a/res18'
    model = mlflow.pytorch.load_model(artifact_uri)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    data_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(data_shape)
    model_traced = torch.jit.trace(model, dummy_input)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    model_path = f"{local_dir}/model.pt"
    model_traced.save(model_path)
    del model
    logger.info("saved the model to model local dir")


@step
def serve_model():
    model_traced = torch.jit.load("model/res18.pt")
    @app.route("/")
    def root():
        return {"message": "Model API is running", "Server":"Uni Cls Respiratory Sys"}

    @app.route("/predict/", methods=["POST"])
    def predict():
        try:
          json = request.json
          if "image" not in json:
              raise ValueError("Bad request image attr not found!")
          input = json['image']
          image = decode_base64_to_pil(input)
          image = image.resize((224, 224))
          image = np.array(image)
          image = process_image(image)
          output = model_traced(image).squeeze()
          output = torch.sigmoid(output)
          return jsonify({"abnormal_prob": output.item()}), 200
        
        except Exception:
            logger.info(str(Exception))  
            return jsonify({"message": "Error in prediction"}), 500
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000)
        