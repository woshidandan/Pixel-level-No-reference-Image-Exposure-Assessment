[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

这是我们组在NIPS 2024关于图像美学评估最新的一篇工作，欢迎Star！: 

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


![Visual](https://github.com/user-attachments/assets/31ba3311-fb0b-4321-bce8-326fc5821354)



# 工作介绍
* 简要版：实现对图像中每个像素曝光观感的判断。
* 太长不看版：各位同仁，若细看图像曝光评估（IEA）这一领域，则不难发现几个问题：1）当前IEA工作的评估细致度不足。例如，在画面同时出现欠曝、过曝及正常曝光时，如何准确评价整体曝光观感？对于局部欠曝，如何量化其程度？显然，传统打分或简单定性分析的方法难以应对此类复杂情况。2）全参考算法在IEA中的精度优于无参考算法，但实际应用中，参考图像往往难以获取，导致无参考IEA算法实用性受限。尽管无参考算法无需参考图像，应用场景灵活，但其评估精度不足，适用性受限。3）现有IEA方法的复用性有限。不同厂商乃至不同场景下的评价标准或规则各异，为某一厂商定制的数据集和算法难以直接应用于另一厂商，增加了二次开发成本。为解决上述问题，我们提出像素级曝光评估的新概念，该评估更侧重于图像美学评估（IAA）而非图像质量评估（IQA）。采用像素级评估的原因如下：1）像素是图像的基本组成单元，从像素维度进行IEA可实现最细粒度的评估，有效应对复杂曝光场景。2）通过像素级标注，可构建理想曝光图像作为监督信息，从而在隐空间中重构待测图像的最佳参考图像，将无参考评估转化为全参考评估。3）像素级标注数据可支持更粗粒度的评估。已知各像素表现后，不同厂商仅需根据各自标准重新映射分数。例如，A厂商规定画面20%过曝扣10分，B厂商则扣30分。在此情况下，本文所述的像素级IEA数据集和算法均可复用。
  

# 网络结构P-IEANet
* 简要版：将图像曝光解耦为亮度和结构信息，分别进行处理，并计算曝光残差（待测图像与理想图像逐像素的差距）。
* 太长不看版：在网络结构设计这块，参考了挺多低光增强和曝光纠正领域的文章，最终结合实际效果（+好说故事）选择采用Haar DWT作为backbone。为什么要从频域将曝光解耦为亮度和结构信息呢？因为我们发现把欠曝和过曝的图像低频分量换一下，欠曝的图像从观感上变成了过曝！
  当然，过曝的图像存在一些死白，导致纹理信息无法恢复，则存在于高频分量中。因此，IEA如果能搞定这两个分量，则评估欠过曝会容易很多。其它一些细节的设计和论证，请参考论文正文和附录把。（注：开源的这版代码有做简化，与原论文略有不同，但训练起来会更容易一些）
![微信截图_20241121163836](https://github.com/user-attachments/assets/c80c05f3-8c85-4248-b1f3-e616e4b69290)


# 代码环境


# 怎么使用代码


# 如果你觉得这篇工作对你有帮助，请引用，不要白嫖-_-:


# 组内其它同类型工作:


# 其它
* 我们实验室的主页：[视觉机器人与智能技术实验室](http://www.mrobotit.cn/Default.aspx)。
* 我的个人主页：[博客](https://xiaohegithub.cn/)，[知乎](https://www.zhihu.com/people/wo-shi-dan-dan-87)。
