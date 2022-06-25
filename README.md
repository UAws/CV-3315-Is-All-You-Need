<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

<br />

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

[📘Documentation](https://mmsegmentation.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) |
[👀Model Zoo](https://mmsegmentation.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmsegmentation.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmsegmentation/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>



---

<h1 id="sec:intro">Background</h1>
<p>Deep learning has been very successful when working with images as
data and is currently at a stage where it works better than humans on
multiple use-cases. The most critical problems humans have been
interested in solving with computer vision are image classification,
object detection and segmentation in the increasing order of their
difficulty. In the plain old image classification task, people are just
interested in getting the labels of all the objects present in an image.
In object detection, researchers come further a step and try to know all
objects in an image and the location at which the objects are present
with the help of bounding boxes. Image segmentation takes it to a new
level by trying to find out accurately the exact boundary of the objects
in the image<span class="citation"
data-cites="surr minaee2021image"></span>.</p>
<p>In this study, we, as beginners in the field of Computer Vision, aim
to develop a basic understanding of semantic segmentation by reviewing,
evaluating, and tuning existing methods, thereby providing a terrific
solution that satisfies both efficiency and accuracy criteria to the
given road segmentation task.</p>
<div align="center" class="figure*">
  <img src="images/Result.png" width="600"/>
</div>
<h2 id="report-structure">Report Structure</h2>
<p>The remaining parts of this report are organized as follows. Section
<a href="#sec:method" data-reference-type="ref"
data-reference="sec:method">2</a> reviews the baseline model and
introduces methods and tricks to be applied. Experiments are conducted
in Section <a href="#sec:experiment" data-reference-type="ref"
data-reference="sec:experiment">3</a>, and Section <a
href="#sec:conclusion" data-reference-type="ref"
data-reference="sec:conclusion">4</a> concludes the report.</p>
<h1 id="sec:method">Method</h1>
<h2 id="baseline-model-analysis">Baseline Model Analysis</h2>

<p>The Baseline model comprises an Encoder-Decoder architecture.
Basically, it extracts the feature maps from the image input and
transforms them into the latent representation. The decoder network then
retrieves those latent-space representations to perform predictions.
Here, the latent-space representation refers to a high channel feature
representation that combines useful underlying semantic information.</p>
<p>The encoder incorporates four downsampling layers to determine the
intermediate features map. During the downsampling process, the
activation function adopts RELU to improve the model’s non-linearity.
MaxPooling plays a significant role during the downsampling operations
for spatial invariance; the pooling layer selects the maximum value of
the current view. The convolutional layers take corresponding input
channels [64, 128, 256, 512, 2048] with kernel size 3x3 and stride 2.
The relatively small kernel decreases the number of parameters and also
enhances the non-linearity; here, the stride is the moving step for the
nearby convolution set to 2 to increase the receptive field.</p>
<p>Also, we found that the baseline network is very similar to FCN,
which is the fundamental work of semantic segmentation.</p>
<p><strong>FCN</strong> <span class="citation" data-cites="fcn"></span>.
Long et al. first proposed using FCNs trained end-to-end for semantic
segmentation. FCN utilizes a skip architecture that integrates semantic
information from a deep, coarse layer with appearance information from a
shallow, fine layer to produce accurate and detailed segmentations. FCNs
have only locally connected layers, such as convolutions, pooling and
upsampling, avoiding any densely connected layer. It also uses skip
connections from its pooling layers to fully recover fine-grained
spatial information lost during downsampling <span class="citation"
data-cites="surr"></span>.</p>

