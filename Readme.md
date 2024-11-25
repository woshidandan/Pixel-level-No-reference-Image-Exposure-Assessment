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


![Visual](https://github.com/user-attachments/assets/31ba3311-fb0b-4321-bce8-326fc5821354)






