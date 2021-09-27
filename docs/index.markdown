---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

## **Abstract**
Disentangling data into interpretable and independent factors is critical
for controllable generation tasks. With the availability of labeled data,
supervision can help enforce the separation of specific factors as expected.
However, it is often expensive or even impossible to label every single
factor to achieve fully-supervised disentanglement. In this paper, we adopt
a general setting where all factors that are hard to label or identify are
encapsulated as a single unknown factor. Under this setting, we propose a
flexible weakly-supervised multi-factor disentanglement framework
*DisUnknown*, which *Dis*tills *Unknown* factors
for enabling multi-conditional generation regarding both labeled and
unknown factors. Specifically, a two-stage training approach is adopted
to first disentangle the unknown factor with an effective and robust
training method, and then train the final generator with the proper
disentanglement of all labeled factors utilizing the unknown distillation.
To demonstrate the generalization capacity and scalability of our method,
we evaluate it on multiple benchmark datasets qualitatively and
quantitatively and further apply it to various real-world applications
on complicated datasets.

## **Architexture**
<center><img src="/figures/train_stage.jpg" width="1152"></center>

## **Application 1:** Anime Style Transfer.
Disentanglement of the artists' identity (*known factor*) and the content 
(*unknown factor*) in anime images,
and its application in anime style transfer.
<center><img src="/figures/app_anime_style_transfer.jpg" width="1152" >
<figcaption>The left column shows four input examples of two different
artists.</figcaption></center>

## **Application 2:** Portrait Relighting
Disentanglement of the lighting (*known factor*) and the remaining content 
(unknown factor) in portrait images,
and its application for portrait relighting.
<center><img src="/figures/app_portrait_relighting.jpg" width="1152" ></center>

## **Application 3:** Landmark-Based Face Reenactment
Disentanglement of the identity (*known factor*), the pose (*known factor*) 
and the expression (*unknown factor*) in 2D face landmarks,
and its application in face reenactment.
<iframe width="1152" height="648" style="margin: auto; display: block;"
src="https://www.youtube.com/embed/BN6LLJpw-v0">
</iframe>

## **Application 4:** Skeleton-Based Body Motion Retargeting
Disentanglement of the identity (*known factor*), the view (*known factor*) 
and the motion (*unknown factor*) in 2D skeletons,
and its application in body motion retargeting.
<center><img src="/figures/app_motion_retargeting.jpg" width="1152" ></center>

## **BibTeX**
<div style="background-color:rgba(0, 0, 0, 0.0470588); text-align:center; vertical-align: middle; padding:30px 60px;">
<p>@article{xiang2021disunknown,</p>
<p>title = {DisUnknown: Distilling Unknown Factors for Disentanglement Learning},</p>
<p>author = {Xiang, Sitao and Gu, Yuming and Xiang, Pengda and Chai, Menglei and Li, 
Hao and Zhao, Yajie and He, Mingming},</p>
<p>journal = {arXiv preprint arXiv:2109.08090},</p>
<p>year = {2021}</p>
<p>}</p>
</div>

