# Datasets

## [DVC](https://dvc.org)
The processed datasets are stored on a Google Drive.  
To use the remote storage edit `.dvc/config.local` (located at the root of the repository) inputting your client ID and secret.   
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


## UCSD Pedestrian
[Download link](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz) (last update: 02/27/2013)

* The folders Peds1 and Peds2 contain the individual frames of each clip in TIFF format.

* Peds1 contains 34 training video samples and 36 testing video samples. 

* Peds2 contains 16 training video samples and 12 testing video samples. 
All testing samples are associated with a manually-collected frame-level abnormal events annotation ground truth list (in the .m file).
   
* The training clips in both sets contain ONLY NORMAL FRAMES. 

* Each of the testing clips contain AT LEAST SOME ANOMALOUS FRAMES. 
A frame-level annotation of abnormal events is provided in the ground truth list under the test folder (in the form of a MATLAB .m file). 
The field 'gt_frame' indicates frames that contain abnormal events.

* 10 test clips from the Peds1 set and 12 from Ped2 are also provided with PIXEL LEVEL GROUNDTRUTH MASKS.
These masks are labeled "Test004_gt", "Test014_gt", "Test018_gt" etc. in the Peds1 folder. 
(There is also full pixel level annotation on Ped1 for all 36 testing videos available at http://hci.iwr.uni-heidelberg.de/COMPVIS/research/abnormality)

* If you use this dataset please cite:

Anomaly Detection in Crowded Scenes.  
V. Mahadevan, W. Li, V. Bhalodia and N. Vasconcelos.  
In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), San Francisco, CA, 2010

* For questions or comments, please contact Weixin Li at wel017@ucsd.edu


## ShanghaiTech Campus
1. Download all split files using [this download link](https://onedrive.live.com/?id=303FB25922AAD438%2173214&cid=303FB25922AAD438)
(from shanghaitech.tar.gz.aa to shanghaitech.tar.gz.ag)

2. Open a Linux or Mac terminal, and merge all split files by 
```shell
cat shanghaitech.tar.gz.* > shanghaitech.tar.gz
```

3. Extract the shanghaitech.tar.gz


## Traffic environment ontology (TE)
- [Homepage](https://spec.edmcouncil.org/auto/ontology/DE/TrafficEnvironment/)
