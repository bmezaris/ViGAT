# ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention networrk

This repository hosts the code and data for our paper: N. Gkalelis, D. Daskalakis, V. Mezaris, "ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention network", IEEE Access vol. XX, no. X, pp. XXX-XXX, month, 2022

## Code requirements

* numpy
* scikit-learn
* PyTorch

## Video preprocessing

Before training Video GAT (ViGAT) on any video dataset, the videos must be preprocessed and converted to an appropriate format for efficient data loading.
Specifically, we sample 9 frames per video for FCVID, 30 frames for miniKinetics and 120 frames per video for ActivityNet.
A global frame feature representation is obtained using a  Vision Transformer (ViT) [6] or ResNet-152 [7] backbone. Moreover, a local frame feature representation is also obtained, i.e., the Faster R-CNN is used as object detector (OD) [4,5] and a feature representation for each object is obtained by applying the network backbone (ViT or ResNet-152) for each object region.
After the video preprocessing stage (i.e. running the Faster-RCNN and network backbone), the dataset root directory must contain the following subdirectories:
* When ViT backbone is used:
  * ```vit_global/```: Numpy arrays of size 9x768 (or 120x768) containing the global frame feature vectors for each video (the 9 (120) frames, times the 768-element vector for each frame).
  * ```vit_local/```: Numpy arrays of size 9x50x768 (or 120x50x768) containing the appearance feature vectors of the detected frame objects for each video (the 9 (120) frames, times the 50 most-prominent objects identified by the object detector, times a 768-element vector for each object bounding box).
* When ResNet-152 backbone is used:
  * ```R152_global/```: Numpy arrays of size 9x2048 (or 120x2048) containing the global frame feature vectors for each video (the 9 (120) frames, times the 2048-element vector for each frame).
  * ```R152_local/```: Numpy arrays of size 9x50x2048 (or 120x50x2048) containing the appearance feature vectors of the detected frame objects for each video (the 9 (120) frames, times the 50 most-prominent objects identified by the object detector, times a 2048-element vector for each object bounding box).

Additionally, the root directory must contain the dataset metadata associated with the dataset:
* For the FCVID, the root directory must contain a ```materials/``` subdirectory with the official training/test split _FCVID\_VideoName\_TrainTestSplit.txt_ and the video event labels _FCVID\_Label.txt_.
* For the ActivityNet, the root directory must contain the officials training/val split _actnet\_train\_split.txt_ and _actnet\_val\_split.txt_
* For the miniKinetics 130k or 85k, the root directory must contain the official training/val split.


## Training

To train a new model end-to-end using the VIT extractor, run
```
python train.py --dataset_root <dataset dir> --dataset [<fcvid|actnet|minikinetics>]
```
By default, the model weights are saved in the ```weights/``` directory. 

This script will also periodically evaluate the performance of the model.

The training parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python train.py --help```.

## Evaluation

To evaluate a model, run
```
python evaluation.py weights/<model name>.pt --dataset_root <dataset dir> --dataset [<fcvid|actnet|minikinetics>]
```
Again, the evaluation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python evaluation.py --help```.

## Explanation

To recreate the metrics for our best model presented in our paper for N frames, run
```
python explanation.py weights/model-actnet-vit-200.pt --dataset_root <dataset dir> --dataset actnet --frames N
```
As previously, the explanation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python explanation.py --help```.


## Usage

To run the code for the different datasets (FCVID, ActivityNet, miniKinetics 135k or 85k) use the corresponding settings described in the paper.
For instance, to train the model end-to-end and evaluate it using the FCVID dataset and ResNet features, run
```
python train.py --dataset_root <FCVID root directory> --dataset fcvid --num_epochs 90 --ext_method RESNET --milestones 30 60 --lr 1e-4 --batch_size 64
```
```
python evaluation.py weights/model-fcvid-resnet-090.pt --dataset_root <FCVID root directory> --dataset fcvid --ext_method RESNET
```

## Provided materials
In this repository, we provide the following models presented in our paper:
* _model-fcvid-resnet-090.pt_ : FCVID model using ResNet-152 FE trained for 90 epochs with initial lr 1e-4 and scheduling at 30 and 60 epochs.
* _model-fcvid-vit-200.pt_ : FCVID model using ViT FE trained for 200 epochs with initial lr 1e-4 and scheduling at 50 and 90 epochs.
* _model-actnet-resnet-090.pt_ : ActivityNet model using ResNet-152 FE trained for 90 epochs with initial lr 1e-4 and scheduling at 30 and 60 epochs.
* _model-actnet-vit-200.pt_ : ActivityNet model using ViT FE trained for 200 epochs with initial lr 1e-4 and scheduling at 110 and 160 epochs.
* _model-minikinetics85k-resnet-090.pt_ : miniKinetics85k model using ResNet-152 FE trained for 90 epochs with initial lr 1e-4 and scheduling at 30 and 60 epochs.
* _model-minikinetics85k-vit-100.pt_ : miniKinetics85k model using ViT FE trained for 100 epochs with initial lr 1e-3 and scheduling at 20 and 50 epochs.
* _model-minikinetics130k-resnet-090.pt_ : miniKinetics130k model using ResNet-152 FE trained for 90 epochs with initial lr 1e-4 and scheduling at 30 and 60 epochs.
* _model-minikinetics130k-vit-100.pt_ : miniKinetics130k model using ViT FE trained for 100 epochs with initial lr 1e-3 and scheduling at 20 and 50 epochs.

Features, bounding boxes and other useful materials extracted during our experiments are provided in the following ftp server:
```
ftp://multimedia2.iti.gr
```
To request access credentials for the ftp please send an email to: bmezaris@iti.gr, gkalelis@iti.gr.

The data stored in the ftp server are:
* FCVID features extracted using ResNet-152 to be placed in the FCVID dataset root directory (~320 GB): FCVID.z01, FCVID.z02, FCVID.z03, FCVID.z04, FCVID.z05, FCVID.z06, FCVID.z07, FCVID.z08, FCVID.z09, FCVID.zip
* FCVID features extracted using ViT to be placed in the FCVID dataset root directory (~110 GB): FCVID_feats.z01, FCVID_feats.z02, FCVID_feats.z03, FCVID_feats.zip
* ActivityNet features extracted using RESNET to be placed in the ActivityNet dataset root directory (~630 GB): ACTNET_feats_RESNET.z01, ACTNET_feats_RESNET.z02, ACTNET_feats_RESNET.z03, ACTNET_feats_RESNET.z04, ACTNET_feats_RESNET.z05, ACTNET_feats_RESNET.z06, ACTNET_feats_RESNET.z07, ACTNET_feats_RESNET.z08, ACTNET_feats_RESNET.z09, ACTNET_feats_RESNET.z10, ACTNET_feats_RESNET.z11, ACTNET_feats_RESNET.z12, ACTNET_feats_RESNET.z13, ACTNET_feats_RESNET.z14, ACTNET_feats_RESNET.z15, ACTNET_feats_RESNET.z16, ACTNET_feats_RESNET.z17, ACTNET_feats_RESNET.zip
* ActivityNet features extracted using ViT to be placed in the ActivityNet dataset root directory (~240 GB): ACTNET_feats.z01, ACTNET_feats.z02, ACTNET_feats.z03, ACTNET_feats.z04, ACTNET_feats.z05, ACTNET_feats.z06, ACTNET_feats.zip
* miniKinetics features extracted using ResNet-152 to be placed in the miniKinetics dataset root directory (934 GB): miniKinetics_feats_RESNET.z01, miniKinetics_feats_RESNET.z02, miniKinetics_feats_RESNET.z03, miniKinetics_feats_RESNET.z04, miniKinetics_feats_RESNET.z05, miniKinetics_feats_RESNET.z06, miniKinetics_feats_RESNET.z07, miniKinetics_feats_RESNET.z08, miniKinetics_feats_RESNET.z09, miniKinetics_feats_RESNET.z10, miniKinetics_feats_RESNET.z11, miniKinetics_feats_RESNET.z12, miniKinetics_feats_RESNET.z13, miniKinetics_feats_RESNET.z14, miniKinetics_feats_RESNET.z15, miniKinetics_feats_RESNET.z16, miniKinetics_feats_RESNET.z17, miniKinetics_feats_RESNET.z18, miniKinetics_feats_RESNET.z19, miniKinetics_feats_RESNET.z20, miniKinetics_feats_RESNET.z21 miniKinetics_feats_RESNET.zip
* miniKinetics features extracted using ViT to be placed in the miniKinetics dataset root directory (358 GB): miniKinetics_feats.z01, miniKinetics_feats.z02, miniKinetics_feats.z03, miniKinetics_feats.z04, miniKinetics_feats.z05, miniKinetics_feats.z06, miniKinetics_feats.z07, miniKinetics_feats.z08, miniKinetics_feats.zip
* FCVID keyframes used, bounding boxes, classes ids and classes scores (~600 MB): FCVID_boxes_etc.zip
* ActivityNet keyframes used, bounding boxes, classes ids and classes scores (~1 GB): ACTNET_boxes_etc.zip
* miniKinetics bounding boxes used, classes ids and classes scores (~6.4 GB): miniKinetics_boxes_etc.zip
Regarding the frames used for the extraction of the provided feats, we extracted 25 frames from each video for FCVID, 30 frames for each video for miniKinetics and used a sampling rate of 1 frame/second for ActivityNet, resulting in around 3 to 6 thousands frames per video. From those, we (randomly) selected 9/30/120 and kept the indices of the selected frames per video in a .txt file, included in the *dataset*_boxes_etc.zip file.
## License and Citation

The code of our ViGAT method is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources (e.g. provided datasets [1,2,3], etc.). If you find the ViGAT code or any of the provided materials useful in your work, please cite the following publication where this approach was proposed:

N. Gkalelis, D. Daskalakis, V. Mezaris, "ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention network", IEEE Access, vol. XX, no. X, pp. XXX, Month, 2022.

Bibtex:
```
@article{ViGAT_Access22,
               author    = "N. Gkalelis and D.Daskalakis and V. Mezaris",
               title     = "ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention network",
               journal   = "IEEE Access",
               year      = "2022",
               volume    = "",
               month     = "",
               number    = "",
               pages     = ""
}
```

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreement 832921 (MIRROR); and, by the QuaLiSID - “Quality of Life Support System for People with Intellectual Disability” project, which is co-financed by the European Union and Greek national funds through the Operational Program Competitiveness, Entrepreneurship and Innovation, under the call RESEARCH-CREATE-INNOVATE (project code: T2EDK-00306).


## References

[1] Y.-G. Jiang, Z. Wu, J. Wang, X. Xue, and S.-F. Chang. Exploiting feature and class relationships in video categorization with regularized deep neural networks. IEEE Trans. Pattern Anal. Mach. Intell., vol. 40, no. 2, pp. 352–364, 2018.

[2] B. G. Fabian Caba Heilbron, Victor Escorcia and J. C. Niebles. ActivityNet: A large-scale video benchmark for human activity understanding. In Proc. IEEE CVPR, 2015, pp. 961–970.

[3] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu and Kevin Murphy. Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification. In Proc. ECCV, 2018, pp. 305-321

[4] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson et al. Bottom-up and top-down attention for image captioning and visual question answering. In Proc. ICVGIP, Hyderabad, India, Dec. 2018, pp. 6077–6086.

[5] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards realtime object detection with region proposal networks. In Proc. NIPS, vol. 28, 2015.

[6] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai et al. An image is worth 16x16 words: Transformers for image recognition at scale. In Proc. ICLR, Virtual Event, Austria, May 2021.

[7] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proc. IEEE/CVF CVPR, Las Vegas, NV, USA, Jun. 2016, pp. 770–778.
