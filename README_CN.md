[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

这是我们组在NIPS 2024关于图像美学评估最新的一篇工作: 

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

因我个人热衷于开源，希望更多的小伙伴关注到这篇工作，故额外写了一篇中文的介绍，不要忘记给我们一个小星星哦，Star一下吧！
------------------------------------------------------------------------------------------------------------

# 工作介绍
* 简要版：实现对图像中每个像素曝光观感的判断
* 太长不看版：各位同仁，若细看图像曝光评估（IEA）这一领域，则不难发现几个问题：1）已有IEA工作的评估粒度不足。
  例如，当画面欠曝、过曝和曝光正常同时存在时，我们应该如何去评价曝光的观感呢？某一个区域存在欠曝，如何去衡量一个区域欠曝的程度呢？
  显然，传统的打分或给出简单定性分析的算法不足以应对这种情况；2）全参考算法的评估精度高于无参考算法，对于IEA也是如此，
  但真实世界中，我们很难找到待测图像的参考图像，因此无参考的IEA算法通常缺乏实用性；相反，无参考算法虽不需要参考图像，使用的场景更加灵活，但其评估的精度不足，缺乏适用性；
  3）已有IEA方法的复用性受限。不同的厂商、甚至不同的场景都有不同的评价标准或规则，当我们历尽千辛为某一个厂商制作了数据集，开发了算法，却无法适应另一个厂商的规则，会导致额外的二次开发成本。
  为了解决上述问题，我们提出了一个新的概念，即像素级的曝光评估，这里的评估，更偏向于于IAA而非IQA，因为衡量的是偏美学的观感。为何要做像素级？1）像素是组成图像的最小单元，若能从这一维度出发进行IEA，
  则能做到最细粒度的评估，能灵活应对欠过曝同时存在的场景；2）通过像素级的标注，实现了对理想曝光图像的构建并作为监督信息，因而能在隐空间中重构待测图像的最佳参考图像，把无参考评估转为全参考评估！
  3）像素级的标注，可以复用到更粗粒度的评估上。举个例子，既然已经知道每个像素的表现情况，每个厂商无非就是做一下重新的打分映射，A厂商：画面存在20%过曝，则扣10分，B厂商：画面存在20%过曝，则扣30分，
  对于这两个厂商而言，原有的像素级IEA数据集和算法都是可以复用的。
  

# 网络结构EAT &nbsp;<a href=""><img width="48" src="https://github.com/woshidandan/Image-Color-Aesthetics-Assessment/assets/15050507/94354c2b-c70e-4d31-bc40-4a2c76d671ff"></a>
* 简要版：通过魔改可变形transformer，解决IAA任务中的注意力偏见问题。
* 太长不看版：若各位同仁，曾对现有的各种IAA网络进行了热力图的可视化，不难发现两个问题，一个是目前的热力图，存在注意力弥散现象，这个在我们去年的[工作](https://github.com/woshidandan/TANet)中有提到，并给了一个
基础的解决办法；还有一个问题，即是本文所关注的注意力偏见问题，即模型只表现出对前景区域的关注。我们是怎么发现这个问题的呢？我们在给甲方交付demo的时候，经常发现，对于一些存在背景虚化的图片评分异常很大，从
人类的角度来说，这种特效还蛮好看的，但对于模型来说，可能就不这么觉得了。另外，我们把AVA数据集内一些背景比较空旷的图像都找出来了，目前的各种IAA模型，在这些样张上表现的效果都比较差。为了解决这个问题，
我们的出发点，是先模拟人类对于图像的关注，并将这种关注以兴趣点形式的最小单元进行表示。但由于目前的IAA模型，均会在ImageNet数据集上进行训练，这些兴趣点还是会优先集中在显著性物体所在的前景区域。为了引导注意
力的方向，我们借助了可变形Transformer中的offset，并对其进行一定的规则限制：探索和利用（做强化学习的同学应该挺熟悉这两个词的）。在网络训练的前期，我们通过计算兴趣点（默认在前景区域）和offset的方向差异，如果
offset奔着兴趣点所在的象限去，则削弱它的趋势，反之，则增强，鼓励网络从非显著性物体所在的背景区域探索更多的美学信息，在训练的后期，则不做什么约束，鼓励网络利用已探索的信息进行美学评分。
* 这套框架性能真的很强，在很多下游的小型IAA任务上表现的都很不错，包括给甲方的基于这套框架改进的demo，在各种牛鬼蛇神的测试场景鲁棒性也较强。这篇工作，也是我个人在IAA赛道上刷SOTA的收官之作，我们也train过一些更SOTA的
版本，但启发性不强。未来会做一些和IAA相关非刷SOTA的，但更有趣的工作！希望各位同行看到我们的工作，审稿时能高抬贵手，ღ( ´･ᴗ･` )比心！

<p align="center">
  <img src="https://github.com/woshidandan/Image-Aesthetics-Assessment/assets/15050507/17a1ea80-7b09-49d4-a85e-bd05464ead82" alt="Image" />
  <img src="https://github.com/woshidandan/Image-Aesthetics-Assessment/assets/15050507/142f495b-0129-4776-bbc7-d808507f643a" alt="Image" />
</p>

# 代码环境


# 怎么使用代码


# 如果你觉得这篇工作对你有帮助，请引用，不要白嫖-_-:


# 组内其它同类型工作:


# 其它
* 我们实验室的主页：[视觉机器人与智能技术实验室](http://www.mrobotit.cn/Default.aspx)。
* 我的个人主页：[博客](https://xiaohegithub.cn/)，[知乎](https://www.zhihu.com/people/wo-shi-dan-dan-87)。
