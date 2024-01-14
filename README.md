# MEDI
Hi, this is the core code of our rencent work "Meta Discovery: Learning to Discover Novel Classes given Very Limited Data" (ICLR 2022, https://openreview.net/pdf?id=MEpKGLsY8f). This work is done by

* Haoang Chi (NUDT), haoangchi618@gmail.com
* Dr. Feng Liu (UTS), feng.liu@uts.edu.au
* Dr. Bo Han (HKBU), bhanml@comp.hkbu.edu.hk
* Dr. Wenjing Yang (NUDT), wenjing.yang@nudt.edu.cn
* Dr. Long Lan (NUDT), long.lan@nudt.edu.cn
* Dr. Tongliang Liu (USYD), tongliang.liu@sydney.edu.au
* Dr. Gang Niu (RIKEN AIP), gang.niu.ml@gmail.com
* Dr. Mingyuan Zhou (UT), mingyuan.zhou@mccombs.utexas.edu
* Prof. Masashi Sugiyama (RIKEN AIP), sugi@k.u-tokyo.ac.jp

# Software
Torch version is 1.7.1. Python version is 3.7.6. CUDA version is 11.0.

These python files, of cause, require some basic scientific computing python packages, e.g., numpy. I recommend users to install python via Anaconda (python 3.7.6), which can be downloaded from https://www.anaconda.com/distribution/#download-section . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda and pytorch (gpu), you can run codes successfully.

# MEDI
Please feel free to test the MEDI method by running main.py.

Specifically, please run

```
CUDA_VISIBLE_DEVICES=0 python main.py
```

in your terminal (using the first GPU device).

The pretrained checkpoints can be downloaded from https://github.com/google-research/simclr?tab=readme-ov-file and converted to Pytorch format with https://github.com/tonylins/simclr-converter.

# Citation
If you are using this code for your own researching, please consider citing

```
@inproceedings{chi2022meta,
  title={Meta discovery: Learning to discover novel classes given very limited data},
  author={Chi, Haoang and Liu, Feng and Han, Bo and Yang, Wenjing and Lan, Long and Liu, Tongliang and Niu, Gang and Zhou, Mingyuan and Sugiyama, Masashi},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

# Acknowledgment
This work was partially supported by the National Natural Science Foundation of China (No.
91948303-1, No. 61803375, No. 12002380, No. 62106278, No. 62101575, No. 61906210)
and the National Grand R&D Plan (Grant No. 2020AAA0103501). BH was supported by NSFC
Young Scientists Fund No. 62006202 and RGC Early Career Scheme No. 22200720. TLL was
supported by Australian Research Council Projects DE-190101473 and DP-220102121. MS was
supported by JST CREST Grant Number JPMJCR18A2.
