# SSCAConv
Source Code and Datasets for "SSCAConv: Self-guided Spatial-Channel Adaptive Convolution for Image Fusion" 

## Source Codes and Datasets for Hyperspectral Pansharpening in Deep Learning


* **Homepage:** [Liang-Jian Deng](https://liangjiandeng.github.io/), [Xiaoya Lu](https://ursulalujun.github.io/about.html) and [Yu-Wei Zhuo](https://pluto-wei.github.io/)
* Code for the paper: "SSCAConv: Self-guided Spatial-Channel Adaptive Convolution for Image Fusion, IEEE GRSL, 2023" [[paper]](https://ieeexplore.ieee.org/document/10367804) 



## Reference

Please cite the related paper:

```bibtex
@ARTICLE{2023lu,
  author={Lu, Xiaoya and Zhuo, Yu-Wei and Chen, Hongming and Deng, Liang-Jian and Hou, Junming},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={SSCAConv: Self-guided Spatial-Channel Adaptive Convolution for Image Fusion}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LGRS.2023.3344944}}
```



## Dependencies and Installation

* Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/))
* Pytorch 1.7.0
* NVIDIA GPU + CUDA
* Python packages: `pip install numpy scipy h5py`
* TensorBoard



## Dataset

* Please find the dataset shared by our team at [[PanCollection]](https://liangjiandeng.github.io/PanCollection.html)
* More info. about these datasets can be found from the published paper.

### Download by ``Baidu Cloud``

* Use [[Baidu Cloud](https://pan.baidu.com/pcloud/home)] to download these datasets.

* ```Pansharpening datasets``` used in this paper are WV3 and GF2. Here we provide complete training and testing data for download.

  | Dataset            | Link                                                         | Password |
  | ------------------ | ------------------------------------------------------------ | -------- |
  | WV3 | [[download link](https://pan.baidu.com/s/1rFf5KdoNp4LakwCNBn-LRQ)] | p6hx     |
  | GF2 | [[download link](https://pan.baidu.com/s/1fhCpNlnLEafkmCFwhc9_zw)] | vbj6     |

* ```HISR task``` Harvard datasets are used for HISR task. Also, we provided the prepared data used in this paper.

  | Dataset            | Link                                                         | Password |
  | ------------------ | ------------------------------------------------------------ | -------- |
  | Harvard       | [[download link](https://pan.baidu.com/s/1JgtKLIcozXec6DfmMs22Cw)] | 8e2i     |

## Code

* data.py: The dataloader of the training and testing data.
* train.py: The main training function of our SSCANet.
* SSCANet.py: The whole model of our SSCANet.
* test.py: The main testing function of our SSCANet.



## Get Started

1. For training, you need to set the file_path in the main function, adopt to your train set, validate set, and test set as well. Our code trains the .h5 file, you may change it through changing the code in data function.
2. After prepareing the dataset, you can modify the model and experiment hyperparameters as needed, such as epoch, learning rate, convergence function, etc. 
3. At the same time, you also need to set the path where the model and log are saved.
4. Then you can start training, the code will automatically save the trained model in .pth format.
5. As for testing, you also need to set the path to open and load the data and trained .pth file, and get the test result in .mat format.




## Method

* Motivation
  
* Proposed module

* Overall Architecture

![image](https://github.com/Pluto-wei/SSCAConv/assets/73097943/8fd33235-1b1a-4084-8ec2-0501e69adea8)


* Visual Results

* Quantitative Results

* Please see the paper for other details.

## Slupplementary experiment resluts
Diffusion model, as a newly developed generative technique, has really drawn much attention. For a more comprehensive and rigorous comparison, we have implemented a newly published diffusion model, i.e., PanDiff [R1], according to your suggestion. Specifically, we compare our SSCANet and PanDiff model on the reduced-resolution WV3 dataset, and the total timesteps T is set to 2000 for PanDiff. Table S1 shows the corresponding quantitative results, in which our model achieves the favorable performance with fewer network parameters and inference time 
compared to PanDiff. Besides, the qualitative visual results are presented in Fig. S1. Due to the 5-page limit 
for this letter, we have provided supplementary experiment results in our online repository.

![image](https://github.com/Pluto-wei/SSCAConv/assets/73097943/cdcb157a-41bc-45fa-8cd1-eb16af80ffd5)

![image](https://github.com/Pluto-wei/SSCAConv/assets/73097943/dc4fe589-28bf-4768-89ad-05854d9efb02)

[R1] Q. Meng, W. Shi, S. Li and L. Zhang, "PanDiff: A Novel Pansharpening Method Based on Denoising 
Diffusion Probabilistic Model," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-17, 
2023, Art no. 5611317, doi: 10.1109/TGRS.2023.3279864.


## Contact

We are glad to hear from you. If you have any questions, please feel free to contact Yuuweii@yeah.net.





