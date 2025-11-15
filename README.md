### CRAD: Cognitive Aware Feature Refinement with Missing Modalities for Early Alzheimer’s Progression Prediction

------------
- This is an official code repository for our paper, which can be found in [here](https://www.sciencedirect.com/science/article/pii/S0895611125001739 "here")👈.
- Hope you like this work and do better works 😁.
- The complete code will be publicly available soon! ⌛
------------

#### Pipeline
[![Consistency Refinement-driven Multi-level Self-Attention Distillation framework. The $\mathcal{N}_T$ learns from paired MRI and PET, whereas $\mathcal{N}_S$ only learns from MRI.](1 "CRAD")](https://ars.els-cdn.com/content/image/1-s2.0-S0895611125001739-gr1_lrg.jpg "CRAD")
The teacher model (N_T) learns from paired MRI and PET, whereas the student model (N_S) only learns from MRI. First, for N_T, the cognitive awareness feature refinement (CAFR) improves the disease-aware ability of the intermediate features by predicting cognitive scores, and the orthogonal projection (OP) disentangles the modality-specific representation to reduce redundancy. Then, the cross-modal self-attention distillation module (PF-CMAD) aligns intermediate features between N_T and N_S, achieving efficient shallow knowledge distillation. Moreover, the smooth distillation unit (SDU) employs the consistency-aware regularization strategy to refine the distillation on intermediate- and high-level features.

------------

#### How to run the code
------------
Below demonstrates how to run this demo code.:stuck_out_tongue_winking_eye:_
1. Create a conda environment <br>
conda create -n CRAD
2. install Pytorch cuda <br>
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
3. install required lib<br>
pip install -r requirements.txt

##### Project structure:
```
CRAD:
	configs:
		config_teacher.py
		config_student.py
	models:
		stas_wgh:
			model.py
	run:
		run_teacher.py
		run_student.py
	utils.py
```

##### Train:

##### Test:


#### Citation
------------
If you think this repository is useful, please cite this paper 😘:
```BibTeX
@article{liu2025crad,
	  title={CRAD: Cognitive Aware Feature Refinement with Missing Modalities for Early Alzheimer’s Progression Prediction},
	  author={Liu, Fei and Liang, Shiuan-Ni and Jaward, Mohamed Hisham and Ong, Huey Fang and Wang, Huabin and Alzheimer’s Disease Neuroimaging Initiative and others},
	  journal={Computerized Medical Imaging and Graphics},
	  pages={102664},
	  year={2025},
	  volumn={126},
	  publisher={Elsevier},
	  doi={doi.org/10.1016/j.compmedimag.2025.102664}
	}
```

#### ### Issue
------------
If you have any questions or problems when using the code in this project, please feel free to send an email to [liu.jason0728@gmail.com](mailto:liu.jason0728@gmail.com "liu.jason0728@gmail.com"). You can also post an issue here.
