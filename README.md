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

#### [DVC](https://dvc.org)
The processed datasets are stored on a Google Drive.  
To use the remote storage edit `.dvc/config.local` inputting your client ID and secret. 
It should look like this:
```
['remote "storage"']
    gdrive_client_id = YOUR_GDRIVE_CLIENT_ID
    gdrive_client_secret = YOUR_GDRIVE_CLIENT_SECRET
```
Then obtain the processed datasets with:
```shell
dcv pull
```

## Model changelog
| Model            | Version | Description                                                                                                                           |
|------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| HSTGCNN          | v1.0.0  | Input: 4 frames<br>Output: 5th frame predictions<br>OA output: L1 + L2 (reduced in temporal dimension)                                |
| KnowledgeHSTGCNN | v1.0.0: | Input: 4 frames + structural ontology features<br>Output: 5th frame predictions<br>OA output: L1 + L2 (reduced in temporal dimension) |
| HSTGCNN          | v2.0.0  | Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>OA output: L1 + L2 + L3 (simplified)                    |
| KnowledgeHSTGCNN | v2.0.0  | Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>OA output: L1 + L2 + L3 (simplified)                    |
| HSTGCNN          | v2.1.0  | Input: 4 frames<br>Output: 4 frames reconstruction + 5th frame predictions<br>OA output: L1 + L2 + L3, softmax(weights)               |
