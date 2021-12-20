# Weight-tied GCNs for event recognition in video

This repository hosts the code and data for our paper: N. Gkalelis, D. Daskalakis, V. Mezaris, "Weight-tied graph convolutional networks  for event recognition and explanation in video", IEEE Transactions on Circuits and Systems for Video Technology, vol. XX, no. X, pp. XXX-XXX, month, 2022

## Code requirements

* numpy
* PyTorch
* sklearn

## Video preprocessing

Before training our method on any video dataset, the videos must be preprocessed and converted to an appropriate format for efficient data loading (in our work, we sample 9 frames per video for FCVID and YLI-MED and 120 frames per video for ActivityNet;
on each frame, a variant of the Faster R-CNN object detector is used [4,5] for object detection and an VIT or ResNet-152 network is used for extracting a representation of each entire frame as well as each object region).
Following video preprocessing, the dataset root directory must contain the following subdirectories:
* For usage of the VIT extractor:
  * ```vit_global/```: Numpy arrays of size 9x768 (or 120x768) containing the global frame feature vectors for each video (the 9 (120) frames, times the 768-element vector for each frame).
  * ```vit_local/```: Numpy arrays of size 9x50x768 (or 120x50x768) containing the appearance feature vectors of the detected frame objects for each video (the 9 (120) frames, times the 50 most-prominent objects identified by the object detector, times a 768-element vector for each object bounding box).
* For usage of the ResNet-152 extractor:
  * ```R152_global/```: Numpy arrays of size 9x2048 (or 120x2048) containing the global frame feature vectors for each video (the 9 (120) frames, times the 2048-element vector for each frame).
  * ```R152_local/```: Numpy arrays of size 9x50x2048 (or 120x50x2048) containing the appearance feature vectors of the detected frame objects for each video (the 9 (120) frames, times the 50 most-prominent objects identified by the object detector, times a 2048-element vector for each object bounding box).

In addition, the root directory must contain the associated dataset metadata:
* The FCVID root directory must contain a ```materials/``` subdirectory with the official training/test split _FCVID\_VideoName\_TrainTestSplit.txt_ and the video event labels _FCVID\_Label.txt_.
* The ActivityNet root directory must contain the officials training/val split _actnet\_train\_split.txt_ and _actnet\_val\_split.txt_
* The YLI-MED root directory must contain the official training/test split _YLI-MED\_Corpus\_v.1.4.txt_.


## Training

To train a new model end-to-end using the VIT extractor, run
```
python train.py --dataset_root <dataset dir> --dataset [<fcvid|actnet|ylimed>]
```
By default, the model weights are saved in the ```weights/``` directory. 

This script will also periodically evaluate the performance of the model.

The training parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python train.py --help```.

## Evaluation

To evaluate a model, run
```
python evaluation.py weights/<model name>.pt --dataset_root <dataset dir> --dataset [<fcvid|actnet|ylimed>]
```
Again, the evaluation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python evaluation.py --help```.

## Explanation

To recreate the metrics for our best model presented in our paper for N frames, run
```
python explanation.py weights/model-actnet-vit-200.pt --dataset_root <dataset dir> --dataset actnet --frames N
```
As previously, the explanation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python explanation.py --help```.


## Usage

To run the code for the different datasets (FCVID, ActivityNet, YLI-MED) use the corresponding settings described in the paper.
For instance, to train the model end-to-end and evaluate it using the FCVID dataset and ResNet features , run
```
python train.py --dataset_root <FCVID root directory> --dataset fcvid --num_epochs 90 --ext_method RESNET --milestones 30 60 --lr 1e-4 --batch_size 64
```
```
python evaluation.py weights/model-fcvid-resnet-090.pt --dataset_root <FCVID root directory> --dataset fcvid --ext_method RESNET
```

## Provided materials
In this repository, we provide the following models presented in our paper:
* _model-fcvid-resnet-090.pt_ : FCVID model using Resnet FE trained for 90 epochs with initial lr 1e-4 and scheduling at 30 and 60 epochs.
* _model-fcvid-vit-200.pt_ : FCVID model using ViT FE trained for 200 epochs with initial lr 1e-4 and scheduling at 50 and 90 epochs.
* _model-actnet-resnet-090.pt_ : ActivityNet model using Resnet FE trained for 90 epochs with initial lr 1e-4 and scheduling at 30 and 60 epochs.
* _model-actnet-vit-200.pt_ : ActivityNet model using ViT FE trained for 200 epochs with initial lr 1e-4 and scheduling at 110 and 160 epochs.
* _model-ylimed-resnet-090.pt_ : YLI-MED model using Resnet FE trained for 90 epochs with initial lr 1e-4 and scheduling at 30 and 60 epochs.
* _model-ylimed-vit-200.pt_ : YLI-MED model using ViT FE trained for 200 epochs with initial lr 1e-3 and scheduling at 110 and 160 epochs.

Features, bounding boxes and other useful materials extracted during our experiments are provided in the following ftp server:
```
ftp://multimedia2.iti.gr
```
To request access credentials for the ftp please send an email to: bmezaris@iti.gr, gkalelis@iti.gr.

The data stored in the ftp server are:
* FCVID features extracted using ResNet-152 to be placed in the FCVID dataset root directory (~320 GB): FCVID.z01, FCVID.z02, FCVID.z03, FCVID.z04, FCVID.z05, FCVID.z06, FCVID.z07, FCVID.z08, FCVID.z09, FCVID.zip
* FCVID features extracted using VIT to be placed in the FCVID dataset root directory (~110 GB): FCVID_feats.z01, FCVID_feats.z02, FCVID_feats.z03, FCVID_feats.zip
* ActivityNet features extracted using RESNET to be placed in the ActivityNet dataset root directory (~630 GB): ACTNET_feats_RESNET.z01, ACTNET_feats_RESNET.z02, ACTNET_feats_RESNET.z03, ACTNET_feats_RESNET.z04, ACTNET_feats_RESNET.z05, ACTNET_feats_RESNET.z06, ACTNET_feats_RESNET.z07, ACTNET_feats_RESNET.z08, ACTNET_feats_RESNET.z09, ACTNET_feats_RESNET.z10, ACTNET_feats_RESNET.z11, ACTNET_feats_RESNET.z12, ACTNET_feats_RESNET.z13, ACTNET_feats_RESNET.z14, ACTNET_feats_RESNET.z15, ACTNET_feats_RESNET.z16, ACTNET_feats_RESNET.z17, ACTNET_feats_RESNET.zip
* ActivityNet features extracted using VIT to be placed in the ActivityNet dataset root directory (~240 GB): ACTNET_feats.z01, ACTNET_feats.z02, ACTNET_feats.z03, ACTNET_feats.z04, ACTNET_feats.z05, ACTNET_feats.z06, ACTNET_feats.zip
* YLI-MED features extracted using ResNet-152 to be placed in the YLI-MED dataset root directory (~6 GB): YLIMED_feats_RESNET.zip
* YLI-MED features extracted using VIT to be placed in the YLI-MED dataset root directory (~3 GB): YLIMED_feats.zip
* FCVID keyframes used, bounding boxes, classes ids and classes scores (~600 MB): FCVID_boxes_etc.zip
* ActivityNet keyframes used, bounding boxes, classes ids and classes scores (~1 GB): ACTNET_boxes_etc.zip
* FCVID keyframes used, bounding boxes, classes ids and classes scores (~14 MB): FCVID_boxes_etc.zip

Regarding the frames used for the extraction of the provided feats, we extracted 25 frames from each video for FCVID and YLI-MED and used a sampling rate of 1 frame/second for ActivityNet, resulting in around 3 to 6 thousands frames per video. From those, we randomly selected 9 (120 for ActivityNet) and kept the indices of the selected frames per video in a .txt file, included in the *dataset*_boxes_etc.zip file.
## License and Citation

The code of our WT-GCN method is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources (e.g. provided datasets [1,2,3], etc.). If you find the WT-GCN code or any of the provided materials useful in your work, please cite the following publication where this approach was proposed:

N. Gkalelis, D. Daskalakis, V. Mezaris, "Weight-tied graph convolutional networks  for event recognition and explanation in video", IEEE Transactions on Circuits and Systems for Video Technology, vol. XX, no. X, pp. XXX, Month, 2022.

Bibtex:
```
@article{WTGCN_CSVT21,
               author    = "N. Gkalelis and D.Daskalakis and V. Mezaris",
               title     = "Weight-tied graph convolutional networks  for event recognition and explanation in video",
               journal   = "IEEE Transactions on Circuits and Systems for Video Technology",
               year      = "2022",
               volume    = "",
               month     = ,
               number    = "",
               pages     = ""
}
```

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreements 832921 (MIRROR).

## References

[1] YY.-G. Jiang, Z. Wu et al. Exploiting feature and class relationships in video categorization with regularized deep neural networks. IEEE Trans. Pattern Anal. Mach. Intell., 40(2):352–364, 2018

[2] B. G. Fabian Caba Heilbron, Victor Escorcia and J. C. Niebles, “Activitynet: A large-scale video benchmark for human activity understanding,”  in Proc. IEEE CVPR, 2015, pp. 961–970.

[3] J. Bernd, D. Borth et al. The YLI-MED corpus: Characteristics, procedures, and plans. CoRR, abs/1503.04250, 2015.

[4] P. Anderson, X. He et al. Bottom-up and top-down attention for image captioning and visual question answering. In Proc. ICVGIP, pages 6077–6086, Hyderabad, India, Dec. 2018

[5] S. Ren, K. He et al. Faster R-CNN: Towards real-time object detection with region proposal networks. In Proc. NIPS, volume 28, 2015.
