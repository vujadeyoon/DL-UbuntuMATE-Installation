# Deep Learning Framework Installation on Ubuntu MATE 18.04 LTS
- Deep Learning Framework Installation on Ubuntu MATE 18.04 LTS
- I recommend that you should ignore the commented instructions with an octothorpe, #.

## Table of contents
0.  [Summarized environments about the DL-UbuntuMATE18.04LTS-Installation](#envs)
1.  [How to set an additional language](#language)
2.  [How to remove the Firefox and install the Opera broweser](#ib)
3.  [How to install a gedit](#gedit)
4.  [How to install and set a Remmina](#remmina)
5.  [How to install and set both ssh and scp](#ssh_scp)
6.  [How to enable a file system, exfat](#exfat)
7.  [How to install a GPU driver](#gpu_driver)
8.  [How to install a CUDA toolkit](#cuda_toolkit)
9.  [How to install a cuDNN](#cudnn)
10. [How to install python 3.7](#python3.7)
11. [How to install and use pip, pip3 and virtualenv](#pip_virtualenv)
12. [How to install and use an Anaconda](#conda)
13. [How to install a PyTorch](#pytorch)
14. [How to install a TensorFlow](#tensorflow)
15. [How to set an Pycharm environment](#pycharm)
16. [Others](#others)


## 0. Summarized environments about the DL-UbuntuMATE18.04LTS-Installation <a name="envs"></a>
- Operating System (OS): Ubuntu MATE 18.04.3 LTS (Bionic)
- Graphics Processing Unit (GPU): NVIDIA TITAN Xp, 1ea
- GPU driver: Nvidia-430.64
- CUDA toolkit: CUDA 10.1
- cuDNN: cuDNN v7.6.5


## 1. How to set an additional language <a name="language"></a>
A. Run the Language Support program and install it completely.<br />
B. Select a fcitx option for a keyboard input method system.<br />
&nbsp; &nbsp; Note that the fcitx option is valid when rerunning the program.<br /> 
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/1.png" width="80%"/><br />
C. Logout and login<br />
D. Click the keyboard icon in the upper right corner of the desktop.<br />
E. Add input method (e.g. Hangul).<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/2.png" width="80%"/><br />
F. Set an input method configuration.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/3.png" width="80%"/><br />


## 2. How to remove the Firefox browser and install the Opera browser <a name="ib"></a>
A. Remove the Firefox browser.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get remove --purge firefox
```
B. Install the Opera browser using a package installer.<br />


## 3. How to install a gedit <a name="gedit"></a>
A. Install the gedit.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install gedit
```


## 4. How to install and set a Remmina <a name="remmina"></a>
A. Reference to the website,
<a href="https://remmina.org" title="Remmina">
Remmina
</a>
.<br />

B. Install the Remmina.<br />
```bash
usrname@hostname:~/curr_path$ sudo snap install remmina
```

C. Set the Remmina remote desktope preference.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/4_Remmina/1.png" width="80%"/><br />


## 5. How to install and set both ssh and scp <a name="ssh_scp"></a>
A. Install the ssh-server.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get remove --purge openssh-server
usrname@hostname:~/curr_path$ sudo apt-get install openssh-server
```


## 6. How to enable a file system, exfat <a name="exfat"></a>
A. Enable the exfat file system.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install exfat-utils exfat-fuse
```


## 7. How to install a GPU driver <a name="gpu_driver"></a>
A. Check a NVIDIA driver version with reference to the website,
<a href="https://www.nvidia.com/Download/Find.aspx" title="NVIDIA driver">
NVIDIA driver
</a>
.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/7_GPU_driver/1.png" width="80%"/><br />

B. Install the NVIDIA driver which user selects.<br />
```bash
usrname@hostname:~/curr_path$ sudo add-apt-repository ppa:graphics-drivers/ppa
usrname@hostname:~/curr_path$ sudo apt-get update
usrname@hostname:~/curr_path$ sudo apt-get install nvidia-driver-430
usrname@hostname:~/curr_path$ sudo reboot
```

C. Check the installed NVIDIA driver version.<br />
```bash
usrname@hostname:~/curr_path$ nvidia-smi
```
```bash
    Mon Jan 27 00:57:44 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 430.64       Driver Version: 430.64       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
    | 23%   33C    P8    13W / 250W |    323MiB / 12192MiB |      1%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      1077      G   /usr/lib/xorg/Xorg                           228MiB |
    +-----------------------------------------------------------------------------+
```

D. Uninstall the installed NVIDIA drive.<br />
```bash
usrname@hostname:~/curr_path$ sudo ./usr/bin/nvidia-uninstall
```


## 8. How to install a CUDA toolkit <a name="cuda_toolkit"></a>
A. Download a CUDA toolkit with reference to the websites,
<a href="https://developer.nvidia.com/cuda-downloads" title="CUDA toolkit">
CUDA toolkit
</a>
and
<a href="https://developer.nvidia.com/cuda-toolkit-archive" title="CUDA toolkit archive">
CUDA toolkit archive
</a>
.<br />
&nbsp; &nbsp; Additional reference to the website, 
<a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#abstract" title="NVIDIA CUDA Installation Guide for Linux">
NVIDIA CUDA Installation Guide for Linux
</a>
.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/8_CUDA_toolkit/10.1.png" width="80%"/><br />

B. Install the CUDA toolkit which user selects.<br />
```bash
usrname@hostname:~/curr_path$ sudo chmod +x cuda_10.1.105_418.39_linux.run
usrname@hostname:~/curr_path$ sudo ./cuda_10.1.105_418.39_linux.run --override
```
```bash
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  End User License Agreement                                                  │
    │  --------------------------                                                  │
    │                                                                              │
    │                                                                              │
    │  Preface                                                                     │
    │  -------                                                                     │
    │                                                                              │
    │  The Software License Agreement in Chapter 1 and the Supplement              │
    │  in Chapter 2 contain license terms and conditions that govern               │
    │  the use of NVIDIA software. By accepting this agreement, you                │
    │  agree to comply with all the terms and conditions applicable                │
    │  to the product(s) included herein.                                          │
    │                                                                              │
    │                                                                              │
    │  NVIDIA Driver                                                               │
    │                                                                              │
    │                                                                              │
    │  Description                                                                 │
    │                                                                              │
    │  This package contains the operating system driver and                       │
    │──────────────────────────────────────────────────────────────────────────────│
    │ Do you accept the above EULA? (accept/decline/quit):                         │
    │ (accept)                                                                     │
    └──────────────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ CUDA Installer                                                               │ 
    │ - [ ] Driver                                                                 │
    │      [ ] 418.39                                                              │
    │ - [X] CUDA Toolkit 10.1                                                      │
    │    + [X] CUDA Tools 10.1                                                     │
    │    + [X] CUDA Libraries 10.1                                                 │
    │    + [X] CUDA Compiler 10.1                                                  │
    │      [X] CUDA Misc Headers 10.1                                              │
    │   [ ] CUDA Samples 10.1                                                      │
    │   [ ] CUDA Demo Suite 10.1                                                   │
    │   [ ] CUDA Documentation 10.1                                                │
    │   Install                                                                    │
    │   Options                                                                    │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
    └──────────────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ CUDA Toolkit                                                                 │
    │   Change Toolkit Install Path                                                │
    │   [X] Create symbolic link from /usr/local/cuda                              │
    │ - [X] Create desktop menu shortcuts                                          │
    │      [X] All users                                                           │
    │      [ ] Yes                                                                 │
    │      [ ] No                                                                  │
    │   [X] Install manpage documents to /usr/share/man                            │
    │   Done                                                                       │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
    └──────────────────────────────────────────────────────────────────────────────┘
```

C. Ignore the below warning about incompleted installation.<br /> 
```bash
    ===========
    = Summary =
    ===========

    Driver:   Not Selected
    Toolkit:  Installed in /usr/local/cuda-10.1/
    Samples:  Not Selected

    Please make sure that
     -   PATH includes /usr/local/cuda-10.1/bin
     -   LD_LIBRARY_PATH includes /usr/local/cuda-10.1/lib64, or, add /usr/local/cuda-10.1/lib64 to /etc/ld.so.conf and run ldconfig as root

    To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-10.1/bin

    Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.1/doc/pdf for detailed information on setting up CUDA.
    ***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 418.00 is required for CUDA 10.1 functionality to work.
    To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
        sudo <CudaInstaller>.run --silent --driver

    Logfile is /var/log/cuda-installer.log
```
```bash
usrname@hostname:~/curr_path$ sudo ./cuda_10.1.105_418.39_linux.run --silent --driver
```

E. Make sure that CUDA path and LD_LIBRARY_path.<br />
```bash
usrname@hostname:~/curr_path$ echo -e "\n## CUDA and cuDNN paths"  >> ~/.bashrc
usrname@hostname:~/curr_path$ echo 'export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
usrname@hostname:~/curr_path$ echo echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
usrname@hostname:~/curr_path$ source ~/.bashrc
usrname@hostname:~/curr_path$ sudo reboot
```

F. Check the installed CUDA toolkit version.<br />
```bash
usrname@hostname:~/curr_path$ nvcc --version
```
```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Fri_Feb__8_19:08:17_PST_2019
    Cuda compilation tools, release 10.1, V10.1.105
```
```bash
usrname@hostname:~/curr_path$ which nvcc
```
```bash
    /usr/local/cuda-10.1/bin/nvcc
```

G. Uninstall the installed CUDA toolkit.<br />
```bash
usrname@hostname:~/curr_path$ sudo ./usr/local/cuda-10.1/bin/uninstall_cuda_10.1.pl
```


## 9. How to install a cuDNN <a name="cudnn"></a>
A. Download a cuDNN (e.g. cuDNN v7.6.5 Library for Linux) with reference to the websites,
<a href="https://developer.nvidia.com/rdp/cudnn-download" title="cuDNN">
cuDNN
</a>
, 
<a href="https://developer.nvidia.com/rdp/cudnn-archive" title="cuDNN archive">
cuDNN archive
</a>
.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/9_cuDNN/7.6.5.png" width="80%"/><br />

B. Install the downloaded cuDNN.<br />
```bash
usrname@hostname:~/curr_path$ tar xzvf cudnn-10.1-linux-x64-v7.6.5.32.tgz
usrname@hostname:~/curr_path$ sudo cp cuda/lib64/* /usr/local/cuda-10.1/lib64/
usrname@hostname:~/curr_path$ sudo cp cuda/include/* /usr/local/cuda-10.1/include/
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.1/include/cudnn.h
```

C. Check the installed cuDNN version.<br />
```bash
usrname@hostname:~/curr_path$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
```bash
    #define CUDNN_MAJOR 7
    #define CUDNN_MINOR 6
    #define CUDNN_PATCHLEVEL 5
    --
    #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

    #include "driver_types.h"
```

D. Install the NVIDIA CUDA profiler tools interface.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install libcupti-dev
```


## 10. How to install python 3.7 <a name="python3.7"></a>
A. Install the python3.7.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get update
usrname@hostname:~/curr_path$ sudo apt-get install software-properties-common
usrname@hostname:~/curr_path$ sudo add-apt-repository ppa:deadsnakes/ppa
usrname@hostname:~/curr_path$ (ENTER)
usrname@hostname:~/curr_path$ sudo apt-get install python3.7
```

C. Check the installed python3.7 version.<br />
```bash
usrname@hostname:~/curr_path$ python3.7 --version
```
```bash
    Python 3.7.6
```
```bash
usrname@hostname:~/curr_path$ python3.7
```
```bash
    Python 3.7.6 (default, Dec 19 2019, 23:50:13) 
    [GCC 7.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.

```


## 11. How to install and use pip, pip3 and virtualenv <a name="pip_virtualenv"></a>
A. Check the pip (pip3) and virtualenv usages with reference to the websites,
<a href="https://pip.pypa.io/en/stable/" title="Pip3">
pip3
</a>
,
<a href="https://virtualenv.pypa.io/en/latest/" title="Virtualenv1">
virtualenv1
</a>
and
<a href="https://packaging.python.org/guides/installing-using-pip-and-virtualenv/" title="Virtualenv2">
virtualenv2
</a>
.<br />

B. Install the pip and pip3.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install python-pip
usrname@hostname:~/curr_path$ sudo apt-get install python3-pip
```

C. Check the installed pip and pip3 version.<br />
```bash
usrname@hostname:~/curr_path$ pip --version
```
```bash
    pip 9.0.1 from /usr/lib/python2.7/dist-packages (python 2.7)
```
```bash
usrname@hostname:~/curr_path$ pip3 --version
```
```bash
    pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)
```

D. Install the virtualenv.<br />
```bash
# usrname@hostname:~/curr_path$ pip3 install virtualenv # This command causes a permission issue on the Ubuntu 18.04.
usrname@hostname:~/curr_path$ sudo pip install virtualenv # You must install the virtualenv as root using the pip, not the pip3.
```
```bash
    Installing collected packages: virtualenv
    Successfully installed virtualenv-16.7.9
```

E. Check the installed virtualenv version.<br />
```bash
usrname@hostname:~/curr_path$ virtualenv --version
```
```bash
    16.7.9
```

F. Create a virtualenv for python 3.7 with pip3.<br />
&nbsp; &nbsp; The root directory for the virtualenv: /home/usrname/pip3_virtualenv<br />
&nbsp; &nbsp; The name of new virtualenv to be created: virenv_dl<br />
```
usrname@hostname:~/curr_path$ mkdir -p /home/usrname/pip3_virtualenv/virenv_dl
usrname@hostname:~/curr_path$ virtualenv /home/usrname/pip3_virtualenv/virenv_dl --python=python3.7
```

G. Check the virtualenv.<br />
&nbsp; &nbsp; The root directory for the virtualenv: /home/usrname/pip3_virtualenv<br />
```bash
usrname@hostname:~/curr_path$ ls /home/usrname/pip3_virtualenv/
```

H. Activate a virtualenv.<br />
&nbsp; &nbsp; The root directory for the virtualenv: /home/usrname/pip3_virtualenv<br />
&nbsp; &nbsp; The name of virtualenv to be activated: virenv_dl<br />
```bash
usrname@hostname:~/curr_path$ source /home/usrname/pip3_virtualenv/virenv_dl/bin/activate
```

I. Deactivate a virtualenv.<br />
&nbsp; &nbsp; The name of virtualenv to be deactivated: virenv_dl<br />
```bash
(virenv_dl) usrname@hostname:~/curr_path$ deactivate
```

J. Remove a virtualenv.<br />
&nbsp; &nbsp; The root directory for the virtualenv: /home/usrname/pip3_virtualenv<br />
&nbsp; &nbsp; The name of virtualenv to be removed: virenv_dl<br />
```bash
(virenv_dl) usrname@hostname:~/curr_path$ deactivate
usrname@hostname:~/curr_path$ rm -rf /home/usrname/pip3_virtualenv/virenv_dl
```

K. Export a pip3 package list.<br />
&nbsp; &nbsp; The name of activated virtualenv: virenv_dl<br />
```bash
(virenv_dl) usrname@hostname:~/curr_path$ pip3 freeze > requirements.txt
```

L. Install packages from the exported pip3 package list.<br />
&nbsp; &nbsp; The name of activated virtualenv: virenv_new<br />
```bash
(virenv_new) usrname@hostname:~/curr_path$ pip3 install -r requirements.txt
```


## 12. How to install and use an Anaconda <a name="conda"></a>
A. Download an Anaconda with reference to the website,
<a href="https://www.anaconda.com/download/#linux" title="Anaconda">
Anaconda
</a>
.<br />

B. Install the downloaded Anaconda.<br />
```bash
usrname@hostname:~/curr_path$ bash Anaconda3-2018.12-Linux-x86_64.sh
```
```bash
    Do you accept the license terms? [yes|no]
    [no] >>> (yes)
    Anaconda3 will now be installed into this location:
    /home/usrname/anaconda3
    [/home/usrname/anaconda3] >>> (ENTER)
    Do you wish the installer to prepend the Anaconda3 install location
    to PATH in your /home/usrname/.bashrc ? [yes|no]
    [no] >>> (yes)
    Do you wish to proceed with the installation of Microsoft VSCode? [yes|no]
    >>> (no)
```
```bash
usrname@hostname:~/curr_path$ source ~/.bashrc
```

<details>
    <summary>C. (Option) Update the Anaconda.</summary>
    
    usrname@hostname:~/curr_path$ conda update conda
    
        The following packages will be UPDATED:
        Proceed ([y]/n)? (y)
        
</details>conda envrionments

D. Check the installed conda version.<br />
```bash
usrname@hostname:~/curr_path$ conda --version
```
```bash
    conda 4.5.12
```

E. Check the conda envrionments.<br />
```bash
usrname@hostname:~/curr_path$ conda info --envs
```
```bash
    # conda environments:
    #
    base                  *  /home/usrname/anaconda3
    
```

F. Create a conda virtual environments for python 3.7 with conda.<br />
&nbsp; &nbsp; The name of new conda virtual environment to be created: conda_dl<br />
```
usrname@hostname:~/curr_path$ conda create --name conda_dl python=3.7
```

G. Clone a conda virtual environment.<br />
&nbsp; &nbsp; The name of new conda virtual environment to be cloned: conda_pytorch<br />
&nbsp; &nbsp; The name of existed conda virtual environment: conda_dl<br />
```bash
usrname@hostname:~/curr_path$ conda create --name conda_pytorch --clone conda_dl
```

H. Activate a conda virtual environment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be activated: conda_pytorch<br />
```bash
usrname@hostname:~/curr_path$ conda activate conda_pytorch
```

I. Deactivate a conda virtual environment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be deactivated: conda_pytorch<br />
```bash
(conda_pytorch) usrname@hostname:~/curr_path$ conda deactivate
```

J. Remove a conda virtual environment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be removed: conda_dl<br />
```bash
usrname@hostname:~/curr_path$ conda remove --name conda_dl --all
```

K. Export a conda package list.<br />
&nbsp; &nbsp; The name of activated conda virtual environment: conda_pytorch<br />
```bash
(conda_pytorch) usrname@hostname:~/curr_path$ conda list -e > requirements.txt
```

L. Install packages from the exported conda package list.<br />
&nbsp; &nbsp; The name of activated conda virtual environment: conda_new<br />
```bash
(conda_new) usrname@hostname:~/curr_path$ conda install --yes --file requirements.txt # does not automatically install all the dependencies
```
```bash
(conda_new) usrname@hostname:~/curr_path$ while read requirement; do conda install --yes $requirement; done < requirements.txt # automatically install all the dependencies
```

M. Export a conda virtual envrionment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be exported: conda_pytorch<br />
&nbsp; &nbsp; The name of exported file: exported_env.yml<br />
```bash
usrname@hostname:~/curr_path$ conda conda_pytorch export > exported_env.yml
```

N. Create a conda virtual environment with the exported conda virtual environment.<br />
&nbsp; &nbsp; The name of new conda virtual environment to be created: conda_new<br />
&nbsp; &nbsp; The name of exported file: exported_env.yml<br />
```bash
usrname@hostname:~/curr_path$ conda conda_new create -f exported_env.yml
```


## 13. How to install a PyTorch <a name="pytorch"></a>
A. Check a PyTorch version with reference to the website,
<a href="https://pytorch.org" title="PyTorch">
PyTorch
</a>
.<br />

B. Install the PyTorch where user want to install it.<br />
&nbsp; &nbsp; The name of virtualenv where user want to install the PyTorch: virenv_pytorch<br />
```bash
usrname@hostname:~/curr_path$ source /home/usrname/pip3_virtualenv/virenv_pytorch/bin/activate
(virenv_pytorch) usrname@hostname:~/curr_path$ pip3 install torch torchvision
```

C. Make sure the PyTorch is installed correctly. <br />
```bash
(virenv_pytorch) usrname@hostname:~/curr_path$ python3
Python 3.7.6 (default, Dec 19 2019, 23:50:13) 
[GCC 7.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```
```python
>>> import torch
>>> torch.__version__
'1.4.0'
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'TITAN Xp'
>>> torch.cuda.device_count()
1
>>> a = torch.rand(5)
>>> b = a.cuda()
>>> print(a)
tensor([0.4732, 0.1292, 0.7363, 0.6000, 0.2162])
>>> print(b)
tensor([0.4732, 0.1292, 0.7363, 0.6000, 0.2162], device='cuda:0')
```


## 14. How to install a TensorFlow <a name="tensorflow"></a>
A. Check a TensorFlow version with reference to the website,
<a href="https://www.tensorflow.org" title="TensorFlow">
TensorFlow
</a>
.<br />

B. Install the TensorFlow where user want to install it.<br />
&nbsp; &nbsp; The name of virtualenv where user want to install the TensorFlow: virenv_tf<br />
```bash
usrname@hostname:~/curr_path$ source /home/usrname/pip3_virtualenv/virenv_tf/bin/activate
(virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow
(virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow-gpu
# The final version of TensorFlow 1.x:
# (virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow==1.15
# (virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow-gpu==1.15
```

C. Make sure the TensorFlow is installed correctly. <br />
```bash
(virenv_tf) usrname@hostname:~/curr_path$ python3
Python 3.7.6 (default, Dec 19 2019, 23:50:13) 
[GCC 7.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```
```python
>>> import tensorflow as tf
2020-01-27 02:02:39.640297: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] ~ please make sure the missing libraries mentioned above are installed properly.
>>> tf.__version__
'2.1.0'
>>> tf.test.is_gpu_available()
WARNING:tensorflow:From <stdin>:1: is_gpu_available ~ GPU (device: 0, name: TITAN Xp, pci bus id: 0000:01:00.0, compute capability: 6.1)
True
>>> tf.debugging.set_log_device_placement(True)
>>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
>>> c = tf.matmul(a, b)
2020-01-27 02:09:26.019321: I tensorflow/core/common_runtime/eager/execute.cc:573] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
2020-01-27 02:09:26.019559: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
>>> print(c)
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```


## 15. How to set an Pycharm environment <a name="pycharm"></a>
A. Download a Pycharm which is a kind of Python IDEs with reference to the website,
<a href="https://www.jetbrains.com/pycharm/download/#section=linux" title="Pycharm">
Pycharm
</a>
.<br />

B. Install the Pycharm.<br />
```bash
usrname@hostname:~/curr_path$ tar xzvf pycharm-community-2018.3.2.tar.gz
usrname@hostname:~/curr_path$ mv pycharm-community-2018.3.2/ ~/
```
I suggest that some options should be selected as follows:
- Complete Installation: Check the option, "Do not import settings".
- Customize PyCharm - Create Launcher Script: Do not check the option.

C. Execute the Pycharm.
```bash
usrname@hostname:~/curr_path$ ~/pycharm-community-2018.3.2/pycharm.sh
```

D. Create a new project with existing interpreter which is in a specific conda environment (e.g. conda_dl).<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/13_Pycharm/1.png" width="80%"/><br />

E. How to set a project interpreter.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/13_Pycharm/2.png" width="80%"/><br />


## 16. Others <a name="others"></a>
A. How to fix NTFS disk write-protect.<br />
```bash
usrname@hostname:~/curr_path$ sudo ntfsfix /dev/sdb1
```

