## Self Supervision to Distillation for Long-Tailed Visual Recognition


  This is a PyTorch implementation of the [SSD-LT](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Self_Supervision_to_Distillation_for_Long-Tailed_Visual_Recognition_ICCV_2021_paper.pdf)


### Requirements
The code is built with following libraries:
- Python==3.6
- PyTorch==1.4.0
- torchvision
- tqdm

### DataSet Preparation

  Download the [ImageNet_2014](https://image-net.org/index). Reorganize the dataset into long-tailed distribution according to image id lists in `./data/`. The directories for the reorganized dataset should look like:

  ```
  |--data
  |--|--train
  |--|--|--n01440764
  |--|--|--|--n01440764_10027.JPEG
  |--|--|--...
  |--|--val
  |--|--|--...
  |--|--test
  |--|--|--...
  ```

### Training

  The training procedure is composed of three stages.
  - Stage I: Self-supervised guided feature learning
    ```
    python ssd_stage_i.py --cos --dist-url 'tcp://localhost:10712' --multiprocessing-distributed --world-size 1 --rank 0 [your imagenet-LT folder]
    ```
  
  - Stage II: Intermediate soft labels generation
    ```
    python ssd_stage_ii.py --cos --last_stage_ckpt 'weights/stage_i/last_checkpoint.pth.tar' --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 [your imagenet-LT folder]
    ```
  
  - Stage III: Joint training with self-distillation
    ```
    python ssd_stage_iii.py --cos --dist-url 'tcp://localhost:11712' --multiprocessing-distributed --world-size 1 --teacher_ckpt 'weights/stage_ii/last_checkpoint.pth.tar' --rank 0 [your imagenet-LT folder]
    ```
  
  An extra classifier fine-tuning step is optional after stage III using `ssd_stage_ii.py` for further improvement.

### Evaluation

An evaluation procedure will be automatically executed when the training is finished. Also, we provide the [last checkpoint](https://drive.google.com/file/d/1z-x-YhOi22SIEGtEYY6hQWKDl_G6jGcw/view?usp=sharing) of stage III for evaluation using the following scripts:
  
  ```
  python ssd_stage_iii.py --dist-url 'tcp://localhost:10712' --multiprocessing-distributed --world-size 1 --rank 0 --resume [your checkpoint path] --evaluate [your imagenet-LT folder]
  ```

The experimental results for stage III on the ImageNet-LT dataset should be like:
|                   | Many| Medium | Few | Overall |
| :---------------: | :---------------: | :---------------: | :---------------: |  :---------------: |
| hard classifier |     71.1 |    46.2    | 15.3 | 51.6 |
| soft classifier | 67.3 | 53.1 | 30.0 | 55.4 |

### Acknowledgements
  We especially thank the contributors of the [Classifier-Balancing](https://github.com/facebookresearch/classifier-balancing) and [MoCo](https://github.com/facebookresearch/moco) for providing helpful code.

### Citation 

  If you think our work is helpful, please feel free to cite our paper.

  ```
  @inproceedings{li2021self,
    title={Self supervision to distillation for long-tailed visual recognition},
    author={Li, Tianhao and Wang, Limin and Wu, Gangshan},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={630--639},
    year={2021}
  }
  ```

### Contact

  For any questions, please feel free to reach `Tianhaolee@outlook.com`.