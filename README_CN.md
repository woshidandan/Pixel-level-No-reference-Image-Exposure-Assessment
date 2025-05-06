[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

🔥这是我们组在NIPS 2024关于图像美学评估最新的一篇工作，欢迎Star！: 

<div align="center">
<h1>
<b>
Rethinking No-reference Image Exposure Assessment from Holism to Pixel: Models, Datasets and Benchmarks
</b>
</h1>
<h4>
<b>
  
Shuai He, Shuntian Zheng, Anlong Ming,  Banyu Wu, Huadong Ma
  
Beijing University of Posts and Telecommunications

University of Warwick

</b>
</h4>
</div>


<div align="center">
<img src="https://github.com/user-attachments/assets/c849349c-bb9a-44bf-a6a0-285b637d4251" alt="Image text" width="300px" />
<img src="https://github.com/user-attachments/assets/72cf4333-cadd-49ee-9e88-aa19aa7e5693" alt="Image text" width="300px" />
<img src="https://github.com/user-attachments/assets/6c7e21d5-8924-484e-a5ce-6053088470c9" alt="Image text" width="300px" />
</div>



## 工作介绍
* 简要版：实现对图像中每个像素曝光观感的判断。
* 太长不看版：各位同仁，若细看图像曝光评估（IEA）这一领域，则不难发现几个问题：
  * 当前IEA工作的评估细致度不足。例如，在画面同时出现欠曝、过曝及正常曝光时，如何准确评价整体曝光观感？对于局部欠曝，如何量化其程度？显然，传统打分或简单定性分析的方法难以应对此类复杂情况。
  * 全参考算法在IEA中的精度优于无参考算法，但实际应用中，参考图像往往难以获取，导致无参考IEA算法实用性受限。尽管无参考算法无需参考图像，应用场景灵活，但其评估精度不足，适用性受限。
  * 现有IEA方法的复用性有限。不同厂商乃至不同场景下的评价标准或规则各异，为某一厂商定制的数据集和算法难以直接应用于另一厂商，增加了二次开发成本。为解决上述问题，我们提出像素级曝光评估的新概念，该评估更侧重于图像美学评估（IAA）而非图像质量评估（IQA）。
* 采用像素级评估的原因如下：
  * 像素是图像的基本组成单元，从像素维度进行IEA可实现最细粒度的评估，有效应对复杂曝光场景。
  * 通过像素级标注，可构建理想曝光图像作为监督信息，从而在隐空间中重构待测图像的最佳参考图像，将无参考评估转化为全参考评估。
  * 像素级标注数据可支持更粗粒度的评估。已知各像素曝光质量后，不同厂商仅需根据各自标准重新映射分数。例如，A厂商规定画面20%过曝扣10分，B厂商则扣30分。在此情况下，本文所述的像素级IEA数据集和算法均可复用。
  

## 网络结构
* 简要版：将图像曝光解耦为亮度和结构信息，分别进行处理，并计算曝光残差（待测图像与理想图像逐像素的差距）。
* 太长不看版：在网络结构设计这块，参考了挺多低光增强和曝光纠正领域的文章，最终结合实际效果（+好说故事）选择采用Haar DWT作为backbone。为什么要从频域将曝光解耦为亮度和结构信息呢？因为我们发现把欠曝和过曝的图像低频分量换一下，欠曝的图像从观感上变成了过曝！
  当然，过曝的图像存在一些死白，导致纹理信息无法恢复，则存在于高频分量中。因此，IEA如果能搞定这两个分量，则评估欠过曝会容易很多。其它一些细节的设计和论证，请参考论文正文和附录把。（注：开源的这版代码有做简化，与原论文略有不同，但训练起来会更容易一些）
<div align="center">
<img src="https://github.com/user-attachments/assets/c80c05f3-8c85-4248-b1f3-e616e4b69290" alt="Image text" width="700px" />
</div>

## 数据集构建
* 构建一个像素级的IEA数据集是极其困难的，逐像素标注不太现实。因此，我们采用了无监督+专家辅助的方式来完成这一任务，其流程如下图所示：
<div align="center">
<img src="https://github.com/user-attachments/assets/835e05e6-7a89-46c8-8889-45a1eece8948" alt="Image text" width="700px" />
</div>
首先，我们通过拍摄或收集（例如Adobe FiveK数据集）一批同一场景不同曝光的图像，对于其中正常曝光的图像使用人工进一步PS，确保其图像所有区域处于理想曝光的状态；其次，将理想图像与同场景曝光欠佳图像相减，得到初始的残差值，但初始的残差值中，会存在零星的标注错误或离散的异常点；最后，专家修正残差值中的错误，形成最终的残差值作为标注信息。
<div align="center">
<img src="https://github.com/user-attachments/assets/dd2ddfdd-f805-4f59-abb4-a977c12b8b09" alt="Image text" width="700px" />
</div>


## 代码使用说明
* 训练目标：给定一张待测图，网络预测其每个像素距离理想曝光图像素的差距，本文称之为残差图（这个是灰度图）；

* 训练：先从[OneDrive](https://bupteducn-my.sharepoint.com/:f:/g/personal/hs19951021_bupt_edu_cn/EleTncDlCnRFnd79v1lzvYYBTTdk7JAe8DtmXWFxIoRjWg?e=e8MbOO)下载数据集（链接挂了cue我），数据集内.npy文件为待测图像对应的残差图，也即网络预测的目标，用train_wavelet.py进行训练；

* 推理：为了测试网络在多幅图像上的预测能力，运行python make_heatmap.py文件，它会读取img_to_pred文件夹中的所有jpg文件进行预测，然后将预测结果的热力图（这个是我们把灰度图进行颜色映射后的图，颜色映射的代码可以自己调整），保存在img_to_pred文件夹中。

<div align="center">
<img src="https://github.com/user-attachments/assets/31ba3311-fb0b-4321-bce8-326fc5821354" alt="Image text" width="700px" />
</div>


## 潜在应用方向
*图像增强算法效果判别器：以低光增强这一超卷方向为例，各同仁在刷到SOTA后，想在一些开放场景与其它同类场景进行效果的横向比较，但开放场景一般没有参考图，所以增强后的效果好不好，只能靠人眼来看，简单定性分析，比如这样：

<div align="center">
<img src="https://github.com/user-attachments/assets/77351a3b-64aa-48bb-8469-56b031a484e7" alt="Image text" width="700px" />
</div>

说实在的，有时候很难看出来区别。我们这项工作，可以作为效果的判别器，既可以通过残差图（或热力图）的形式，更加直观的分析各算法增强后效果的优劣，比如下图，越黑代表增强效果越好，也能把残差图转为数值（求个MAE），进行定量的分析。

<div align="center">
<img src="https://github.com/user-attachments/assets/0bc7ae1f-acc8-4041-9d53-5192e5b14c92" alt="Image text" width="700px" />
</div>

*图像增强算法增强器：没错，我们这项工作也能作为loss函数，集成到一些低光增强或曝光纠正算法里，作为一个reward辅助模型训练出更好的视觉效果，甚至能提升模型性能：

<div align="center">

<img src="https://github.com/user-attachments/assets/2935704b-a972-428e-b64f-ee7d6e23cf32" alt="Image text" width="700px" />

</div>

## 代码环境
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

## 如果你觉得这篇工作对你有帮助，请引用，不要白嫖-_-:
```
@article{herethinkingIEA,
  title={Rethinking No-reference Image Exposure Assessment from Holism to Pixel: Models, Datasets and Benchmarks},
  author={He, Shuai and Zheng, Shuntian and Ming, Anlong and Wu, Banyu and Ma, Huadong},
  journal={Advances in Neural Information Processing Systems (NIPS)},
  year={2024},
}
```

## 组内其它同类型工作:
<table>
  <thead align="center">
    <tr>
      <td><b>🎁 Projects</b></td>
      <td><b>📚 Publication</b></td>
      <td><b>🌈 Content</b></td>
      <td><b>⭐ Stars</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment"><b>Pixel-level image exposure assessment【首个像素级曝光评估】</b></a></td>
      <td><b>NIPS 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment"><b>Long-tail solution for image aesthetics assessment【美学评估数据不平衡解决方案】</b></a></td>
      <td><b>ICML 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Prompt-DeT"><b>CLIP-based image aesthetics assessment【基于CLIP多因素色彩美学评估】</b></a></td>
      <td><b>Information Fusion 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Prompt-DeT?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment"><b>Compare-based image aesthetics assessment【基于对比学习的多因素美学评估】</b></a></td>
      <td><b>ACMMM 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment"><b>Image color aesthetics assessment【首个色彩美学评估】</b></a></td>
      <td><b>ICCV 2023</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Aesthetics-and-Quality-Assessment"><b>Image aesthetics assessment【通用美学评估】</b></a></td>
      <td><b>ACMMM 2023</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment"><b>Theme-oriented image aesthetics assessment【首个多主题美学评估】</b></a></td>
      <td><b>IJCAI 2022</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/TANet-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/AK4Prompts"><b>Select prompt based on image aesthetics assessment【基于美学评估的提示词筛选】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/AK4Prompts?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/mRobotit/M2Beats"><b>Motion rhythm synchronization with beats【动作与韵律对齐】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/mRobotit/M2Beats?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC"><b>Champion Solution for AIGC Image Quality Assessment【NTIRE AIGC图像质量评估赛道冠军】</b></a></td>
      <td><b>CVPRW NTIRE 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC?style=flat-square&labelColor=343b41"/></td>
    </tr>
  </tbody>
</table>

# 其它
* 我们实验室的主页：[视觉机器人与智能技术实验室](http://www.mrobotit.cn/Default.aspx)。
* 我的个人主页：[博客](https://xiaohegithub.cn/)，[知乎](https://www.zhihu.com/people/wo-shi-dan-dan-87)。
