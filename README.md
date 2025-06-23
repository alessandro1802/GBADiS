# Graph-Based Anomaly Detection in Surveillance
This repository contains source code of experiments for a MSc thesis at Poznań University of Technology.

## Set-up
### Project dependecies
```shell
python3.11 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

### Submodules
- [YOLOv5](https://github.com/ultralytics/yolov5): 
`yolov5x` weights can be downloaded from [this URL](https://github.com/ultralytics/yolov5#pretrained-checkpoints) and should be put under `src/yolov5/weights/`

- [HRnet](https://github.com/HRNet/HRNet-Human-Pose-Estimation): 
`pose_hrnet_w48_384x288` weights are available at [this link](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA) and should be put under `src/hrnet/weights/`   
Install `geos` dependency with, e.g.:
```shell
brew install geos
```

- [RAFT](https://github.com/princeton-vl/RAFT): 
`raft-sintel` weights can be fetched from [this URL](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) and should be put under `src/raft/weights/`

### MLOps
#### [MLflow](https://mlflow.org)
Spin-up a local instance on `http://localhost:5050` using [Docker](https://www.docker.com):
```shell
docker run -d \
  --name mlflow-thesis \
  -p 5050:5000 \
  -v $(pwd):$(pwd) \
  ghcr.io/mlflow/mlflow \
  mlflow server \
    --backend-store-uri $(pwd)/mlruns \
    --default-artifact-root file:$(pwd)/mlruns \
    --host 0.0.0.0 \
    --port 5000
```
Copy `.env.example` to `.env` and put your `MLFLOW_TRACKING_URI` address, e.g., `"http://localhost:5050"`.  

#### [DVC](https://dvc.org)
Refer to `data/README.md` for description and instructions.

## Model changelog
| Model            | Version | Description                                                                                                                                                                                                                                    |
|------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HSTGCNN          | v1.0.0  | Input: 4 frames<br>Output: 5th frame predictions<br>OA output: L1 + L2 (reduced in temporal dimension)                                                                                                                                         |
| KnowledgeHSTGCNN | v1.0.0  | Input: 4 frames + structural ontology features<br>Output: 5th frame predictions<br>OA output: L1 + L2 (reduced in temporal dimension)                                                                                                          |
| HSTGCNN          | v2.0.0  | Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>OA output: L1 + L2 + L3 (simplified)                                                                                                                             |
| KnowledgeHSTGCNN | v2.0.0  | Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>OA output: L1 + L2 + L3 (simplified)                                                                                                                             |
| HSTGCNN          | v2.1.0  | Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>OA output: L1 + L2 + L3, softmax(weights)                                                                                                                        |
| KnowledgeHSTGCNN | v3.0.0  | Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>Feature fusion updated with ontology node embeddings multi-head cross-attention mechanism<br>OA output: L1 + L2 + L3, softmax(weights)                           |
| HSTGCNN          | v2.5.0  | Trained on new dataset<br>Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>OA output: L1 + L2 + L3, softmax(weights)                                                                                              |

## References
- HSTGCNN: Zeng et al. (2021). A hierarchical spatio-temporal graph convolutional neural network for anomaly detection in videos. **IEEE TCSVT**, 33(1), 200-212.
