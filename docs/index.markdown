---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---
Contents are still being updated. You may check the code first.

## Abstract
Disentangling data into interpretable and independent factors is critical
for controllable generation tasks. With the availability of labeled data,
supervision can help enforce the separation of specific factors as expected.
However, it is often expensive or even impossible to label every single
factor to achieve fully-supervised disentanglement. In this paper, we adopt
a general setting where all factors that are hard to label or identify are
encapsulated as a single unknown factor. Under this setting, we propose a
flexible weakly-supervised multi-factor disentanglement framework
**DisUnknown**, which **Dis**tills **Unknown** factors
for enabling multi-conditional generation regarding both labeled and
unknown factors. Specifically, a two-stage training approach is adopted
to first disentangle the unknown factor with an effective and robust
training method, and then train the final generator with the proper
disentanglement of all labeled factors utilizing the unknown distillation.
To demonstrate the generalization capacity and scalability of our method,
we evaluate it on multiple benchmark datasets qualitatively and
quantitatively and further apply it to various real-world applications
on complicated datasets.

## The Paper
[Read on arXiv](https://arxiv.org/abs/2109.08090)

## Videos
Disentanglement of identity and pose in 2D face landmarks,
and its application in face reenactment.
<iframe width="640" height="360" style="margin: auto; display: block;"
src="https://www.youtube.com/embed/BN6LLJpw-v0">
</iframe>