[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
Rethinking No-reference Image Exposure Assessment from Holism to Pixel:  Models, Datasets and Benchmarks
</b>
</h1>
<h4>
<b>
Shuai He, Shuntian Zheng, Anlong Ming, Banyu Wu, Huadong Ma
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

[[国内的小伙伴请看更详细的中文说明]](https://github.com/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment/blob/main/README_CN.md)
This repo contains the official implementation and the new dataset IEA40K of the **NIPS 2024** paper.
More details will be released soon.

<div align="center">
<img src="https://github.com/user-attachments/assets/c849349c-bb9a-44bf-a6a0-285b637d4251" alt="Image text" width="280px" />
<img src="https://github.com/user-attachments/assets/72cf4333-cadd-49ee-9e88-aa19aa7e5693" alt="Image text" width="280px" />
<img src="https://github.com/user-attachments/assets/6c7e21d5-8924-484e-a5ce-6053088470c9" alt="Image text" width="280px" />
</div>

## Introduction
* **Brief Version**: Achieving the evaluation of exposure quality for every pixel in an image.

* **DR Version**: Colleagues, upon delving into the field of Image Exposure Assessment (IEA), several issues become apparent:

  * Current IEA methods lack sufficient granularity. For instance, how can we accurately evaluate the overall exposure perception when underexposed, overexposed, and normally exposed areas coexist in an image? How can we quantify the degree of local underexposure? Clearly, traditional scoring or simple qualitative analysis cannot address such complex scenarios.

  * Full-reference algorithms outperform no-reference algorithms in IEA precision. However, in practical applications, reference images are often unavailable, limiting the usability of no-reference algorithms. While no-reference algorithms do not require reference images and are more flexible in application, their evaluation accuracy and general applicability remain insufficient.

  * The reusability of existing IEA methods is limited. Evaluation standards or rules vary across manufacturers and scenarios, making datasets and algorithms tailored for one manufacturer difficult to apply to another, thereby increasing redevelopment costs. 

To address these issues, we propose a new concept of pixel-level IEA. This approach focuses more on Image Aesthetic Assessment (IAA) rather than Image Quality Assessment (IQA).

* **Reasons for Adopting Pixel-Level Evaluation**:
  * Pixels are the fundamental units of an image. Conducting IEA at the pixel level enables the finest-grained evaluation, effectively handling complex exposure scenarios.
    
  * Through pixel-level annotations, it is possible to construct an ideal exposure image as supervision information. This allows the reconstruction of the optimal reference image for the target image in latent space, thereby transforming no-reference evaluation into full-reference evaluation.
    
  * Pixel-level annotated data can support coarser-grained assessments. Once the exposure quality of each pixel is known, manufacturers can simply remap the scores according to their standards. For example, Manufacturer A might deduct 10 points for 20% overexposure in the image, while Manufacturer B might deduct 30 points. Under such circumstances, the pixel-level IEA datasets and algorithms described in this paper can be reused effectively.

<div align="center">
<img src="https://github.com/user-attachments/assets/31ba3311-fb0b-4321-bce8-326fc5821354" alt="Image text" width="700px" />
</div>

## Network
* **Brief Version**: Decouple image exposure into brightness and structural information, process them separately, and compute the exposure residual (pixel-wise difference between the test image and the ideal image).

* **DR Version**: 
  In terms of network architecture design, we referred to several works in the fields of low-light enhancement and exposure correction. Ultimately, based on practical performance (and to build a coherent narrative), we chose Haar DWT as the backbone. 
  Why decouple exposure into brightness and structural information in the frequency domain? Because we discovered that swapping the low-frequency components of underexposed and overexposed images makes the underexposed image appear overexposed from a perceptual perspective! 
  Of course, overexposed images often have "dead white" areas where texture information cannot be recovered, and this issue resides in the high-frequency components. Therefore, if IEA can address these two components, evaluating underexposure and overexposure becomes much simpler. 
  For more details on the design and justification, please refer to the paper. (Note: the open-source version of the code is slightly simplified compared to the original paper to make training easier.)

<div align="center">
<img src="https://github.com/user-attachments/assets/c80c05f3-8c85-4248-b1f3-e616e4b69290" alt="Image text" width="700px" />
</div>

## Dataset
Constructing a pixel-level IEA dataset is extremely challenging, as pixel-by-pixel annotation is impractical. Therefore, we adopted an unsupervised approach combined with expert assistance to accomplish this task. The workflow is illustrated in the figure below:

<div align="center">
<img src="https://github.com/user-attachments/assets/835e05e6-7a89-46c8-8889-45a1eece8948" alt="Image text" width="700px" />
</div>

First, we captured or collected (e.g., from the Adobe FiveK dataset) a set of images of the same scene under different exposure conditions. For the normally exposed images, we further refined them manually in Photoshop to ensure all regions in the images are in an ideal exposure state. 
Next, we subtracted the ideal images from their poorly exposed counterparts of the same scene to obtain the initial residual values. However, these initial residuals might contain sporadic annotation errors or discrete outliers. 
Finally, experts corrected the errors in the residual values to generate the final residuals, which serve as the annotation data.

<div align="center">
<img src="https://github.com/user-attachments/assets/dd2ddfdd-f805-4f59-abb4-a977c12b8b09" alt="Image text" width="700px" />
</div>

## Code Usage Instructions
* ### **Training Objective**
The goal is to predict the residual map (a grayscale image) for each pixel of a test image, representing the pixel-wise difference between the test image and the ideal exposure image.

* ### **Training Steps**
1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1zZPRgHvhr6OTr-wuhJYcs8H2DOVtL62Y/view) (if the link is broken, let me know).
2. The `.npy` files in the dataset represent the residual maps corresponding to the test images, which are the target outputs for the network.
3. Use `train_wavelet.py` to train the network.

* ### **Inference**
1. To test the network's prediction ability on multiple images, run the script `make_heatmap.py`.
2. The script will read all `.jpg` files in the `img_to_pred` folder for prediction.
3. The predicted heatmaps (color-mapped versions of the grayscale residual maps) will be saved in the same `img_to_pred` folder. You can customize the color-mapping code as needed.

<div align="center">
<img src="https://github.com/user-attachments/assets/31ba3311-fb0b-4321-bce8-326fc5821354" alt="Image text" width="700px" />
</div>

## Potential Application Directions

### **1. Evaluator for Image Enhancement Algorithms**
Taking low-light enhancement as an example within this trending field, researchers often seek to perform cross-comparisons of enhancement results across various scenarios after achieving SOTA (state-of-the-art) results. However, open scenarios usually lack reference images, leaving the assessment of enhancement quality to human visual judgment or simple qualitative analyses. For instance:

<div align="center">
<img src="https://github.com/user-attachments/assets/77351a3b-64aa-48bb-8469-56b031a484e7" alt="Image text" width="700px" />
</div>

In practice, it can be difficult to distinguish differences. Our method serves as an evaluator, providing a more intuitive analysis of enhancement performance through residual maps (or heatmaps). For example, in the figure below, darker areas indicate better enhancement performance. Additionally, residual maps can be converted into numerical values (e.g., by calculating MAE) for quantitative analysis.

<div align="center">
<img src="https://github.com/user-attachments/assets/0bc7ae1f-acc8-4041-9d53-5192e5b14c92" alt="Image text" width="700px" />
</div>

---

### **2. Enhancer for Image Enhancement Algorithms**
Our work can also be used as a loss function, integrated into low-light enhancement or exposure correction algorithms. It can act as a reward, helping to train models for better visual effects and even improve their performance:

<div align="center">
<img src="https://github.com/user-attachments/assets/2935704b-a972-428e-b64f-ee7d6e23cf32" alt="Image text" width="700px" />
</div>

## Environment Installation
```
* tqdm==4.66.2
* torchvision==0.16.2+cu121
* torchsummary==1.5.1
* torch==2.1.2+cu121
* timm==0.6.13
* tensorboard==2.14.0
* scipy==1.10.1
* scikit-learn==1.3.2
* PyYAML==6.0.1
* pytorch-ssim==0.1
* pytorch-msssim==1.0.0
* pillow==10.2.0
* pandas==2.0.3
* opencv-python==4.9.0.80
* numpy==1.24.4
* matplotlib==3.7.5
```

## If you find our work is useful, pleaes cite our paper:
```
@article{herethinkingIEA,
  title={Rethinking No-reference Image Exposure Assessment from Holism to Pixel: Models, Datasets and Benchmarks},
  author={He, Shuai and Zheng, Shuntian and Ming, Anlong and Wu, Banyu and Ma, Huadong},
  journal={Advances in Neural Information Processing Systems (NIPS)},
  year={2024},
}
```

## Related Work from Our Group

- **"Thinking Image Color Aesthetics Assessment: Models, Datasets and Benchmarks."**  
  [[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/He_Thinking_Image_Color_Aesthetics_Assessment_Models_Datasets_and_Benchmarks_ICCV_2023_paper.pdf)  
  [[Code]](https://github.com/woshidandan/Image-Color-Aesthetics-Assessment)  
  *ICCV 2023*  

- **"Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks."**  
  [[PDF]](https://www.ijcai.org/proceedings/2022/0132.pdf)  
  [[Code]](https://github.com/woshidandan/TANet)  
  *IJCAI 2022*  

- **"EAT: An Enhancer for Aesthetics-Oriented Transformers."**  
  [[PDF]](https://github.com/woshidandan/Image-Aesthetics-Assessment/blob/main/Paper_ID_847_EAT%20An%20Enhancer%20for%20Aesthetics-Oriented%20Transformers.pdf)  
  [[Code]](https://github.com/woshidandan/Image-Aesthetics-Assessment/tree/main)  
  *ACMMM 2023*  

---

{% include https://github.com/woshidandan/woshidandan/edit/master/README.md %}


## Additional Information

- **Lab Homepage**: [Visual Robotics and Intelligent Technology Laboratory](http://www.mrobotit.cn/Default.aspx)  
- **My Personal Pages**:  
  - [Blog](https://xiaohegithub.cn/)  
  - [Zhihu](https://www.zhihu.com/people/wo-shi-dan-dan-87)  
