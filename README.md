# ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention network

This repository hosts the code and data for our paper: N. Gkalelis, D. Daskalakis, V. Mezaris, "ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention network", IEEE Access, 2022. DOI: 10.1109/ACCESS.2022.3213652.

## ViGAT scripts, and traning and evaluation procedures

### Code requirements

* numpy
* scikit-learn
* PyTorch

### Video preprocessing

Before training Video GAT (ViGAT) on any video dataset, the videos must be preprocessed and converted to an appropriate format for efficient data loading.
Specifically, we sample 9 frames per video for FCVID, 30 frames for MiniKinetics and 120 frames per video for ActivityNet.
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
* For the MiniKinetics 130k or 85k, the root directory must contain the official training/val split.


### Training

To train a new model end-to-end using the VIT extractor, run
```
python train.py --dataset_root <dataset dir> --dataset [<fcvid|actnet|minikinetics>]
```
By default, the model weights are saved in the ```weights/``` directory. 

This script will also periodically evaluate the performance of the model.

The training parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python train.py --help```.

### Evaluation

To evaluate a model, run
```
python evaluation.py weights/<model name>.pt --dataset_root <dataset dir> --dataset [<fcvid|actnet|minikinetics>]
```
Again, the evaluation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python evaluation.py --help```.

### Explanation

To recreate the metrics for our best model presented in our paper for N frames, run
```
python explanation.py weights/model-actnet-vit-200.pt --dataset_root <dataset dir> --dataset actnet --frames N
```
As previously, the explanation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python explanation.py --help```.


### Usage

To run the code for the different datasets (FCVID, ActivityNet, MiniKinetics 135k or 85k) use the corresponding settings described in the paper.
For instance, to train the model end-to-end and evaluate it using the FCVID dataset and ResNet features, run
```
python train.py --dataset_root <FCVID root directory> --dataset fcvid --num_epochs 90 --ext_method RESNET --milestones 30 60 --lr 1e-4 --batch_size 64
```
```
python evaluation.py weights/model-fcvid-resnet-090.pt --dataset_root <FCVID root directory> --dataset fcvid --ext_method RESNET
```

## Additional materials: extracted features and trained ViGAT models

In this repository, we provide the models that use ViT features, as presented in our paper:

* _model-fcvid-vit-200.pt_ : FCVID model using ViT FE trained for 200 epochs with initial lr 1e-4 and scheduling at 50 and 90 epochs.
* _model-actnet-vit-200.pt_ : ActivityNet model using ViT FE trained for 200 epochs with initial lr 1e-4 and scheduling at 110 and 160 epochs.
* _model-minikinetics_85k-vit-100.pt_ : MiniKinetics85k model using ViT FE trained for 100 epochs with initial lr 1e-3 and scheduling at 20 and 50 epochs.
* _model-minikinetics_130k-vit-100.pt_ : MiniKinetics130k model using ViT FE trained for 100 epochs with initial lr 1e-3 and scheduling at 20 and 50 epochs.

Due to file size limitations here, additional models trained using ResNet features, as well as features, bounding boxes and other useful materials extracted during our experiments, are available on an ftp server. To request access credentials for the ftp please send an email to: bmezaris@iti.gr, gkalelis@iti.gr.

The additional data stored in the ftp server are:

* _model-fcvid-resnet-090.pt_ (~300 MB)
* _model-actnet-resnet-090.pt_ (~300 MB)
* _model-minikinetics_85k-resnet-090.pt_ (~300 MB)
* _model-minikinetics_130k-resnet-090.pt_ (~300 MB)
* FCVID features extracted using ResNet-152 to be placed in the FCVID dataset root directory (~320 GB)
* FCVID features extracted using ViT to be placed in the FCVID dataset root directory (~110 GB)
* ActivityNet features extracted using RESNET to be placed in the ActivityNet dataset root directory (~630 GB)
* ActivityNet features extracted using ViT to be placed in the ActivityNet dataset root directory (~240 GB)
* MiniKinetics_85k features extracted using ResNet-152 to be placed in the MiniKinetics dataset root directory (934 GB)
* MiniKinetics_85k features extracted using ViT to be placed in the MiniKinetics dataset root directory (358 GB)
* FCVID keyframes used, bounding boxes, classes ids and classes scores (~600 MB)
* ActivityNet keyframes used, bounding boxes, classes ids and classes scores (~1 GB)
* MiniKinetics bounding boxes used, classes ids and classes scores (~6.4 GB)

Regarding the frames used for the extraction of the provided feats, for FCVID we extracted 25 frames and selected (randomly) 9 frames per video, while, for ActivityNet we used a sampling rate of 1 frame/second, resulting to approximately 6 thousands frames per video, and selected 120 frames per video. The frame indices of the selected frames were saved in a .txt file, included in the *dataset*\_boxes_etc.zip file.  Similarly, for MiniKinetics85K we extracted 30 frames per video, and for MiniKinetics130K 120 frames and selected 30 frames per video. 

## License

This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources (e.g. provided datasets [1,2,3], etc.). Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation

If you find the ViGAT code or any of the provided materials useful in your work, please cite the following publication where this approach was proposed:

N. Gkalelis, D. Daskalakis, V. Mezaris, "ViGAT: Bottom-up event recognition and explanation in video using factorized graph attention network", IEEE Access, 2022. DOI: 10.1109/ACCESS.2022.3213652.

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
               pages     = "",
               doi       = {10.1109/ACCESS.2022.3213652},
               url       = {https://doi.org/10.1109/ACCESS.2022.3213652}        
}
```

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreement 832921 (MIRROR) and 101021866 (CRiTERIA); and, by the QuaLiSID - “Quality of Life Support System for People with Intellectual Disability” project, which is co-financed by the European Union and Greek national funds through the Operational Program Competitiveness, Entrepreneurship and Innovation, under the call RESEARCH-CREATE-INNOVATE (project code: T2EDK-00306).


## References

[1] Y.-G. Jiang, Z. Wu, J. Wang, X. Xue, and S.-F. Chang. Exploiting feature and class relationships in video categorization with regularized deep neural networks. IEEE Trans. Pattern Anal. Mach. Intell., vol. 40, no. 2, pp. 352–364, 2018.

[2] B. G. Fabian Caba Heilbron, Victor Escorcia and J. C. Niebles. ActivityNet: A large-scale video benchmark for human activity understanding. In Proc. IEEE CVPR, 2015, pp. 961–970.

[3] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu and Kevin Murphy. Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification. In Proc. ECCV, 2018, pp. 305-321

[4] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson et al. Bottom-up and top-down attention for image captioning and visual question answering. In Proc. ICVGIP, Hyderabad, India, Dec. 2018, pp. 6077–6086.

[5] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards realtime object detection with region proposal networks. In Proc. NIPS, vol. 28, 2015.

[6] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai et al. An image is worth 16x16 words: Transformers for image recognition at scale. In Proc. ICLR, Virtual Event, Austria, May 2021.

[7] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proc. IEEE/CVF CVPR, Las Vegas, NV, USA, Jun. 2016, pp. 770–778.
