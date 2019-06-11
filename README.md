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
10. [How to install and use pip3 and virtualenv](#pip3_virtualenv)
11. [How to install and use an Anaconda](#conda)
12. [How to install a PyTorch](#pytorch)
13. [How to install a TensorFlow](#tensorflow)
14. [How to set an Pycharm environment](#pycharm)
15. [Others](#others)


## 0. Summarized environments about the DL-UbuntuMATE18.04LTS-Installation <a name="envs"></a>
- Operating System (OS): Ubuntu MATE 18.04.1 LTS (Bionic)
- Graphics Processing Unit (GPU): NVIDIA TITAN Xp
- GPU driver: Nvidia-410.78
- CUDA toolkit: CUDA 10.0
- cuDNN: cuDNN v7.4.2


## 1. How to set an additional language <a name="language"></a>
A. Run the Language Support program and install it completely.<br />
B. Select a fcitx option for a keyboard input method system.<br />
&nbsp; &nbsp; Note that the fcitx option is valid when rerunning the program.<br /> 
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/1.png" width="80%"/><br />
C. Logout and login<br />
D. Add input method (e.g. Hangul)<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/2.png" width="80%"/><br />
E. Set an input method configuration.<br />
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
# usrname@hostname:~/curr_path$ sudo apt-get install nvidia-390
usrname@hostname:~/curr_path$ sudo apt-get install nvidia-driver-410
usrname@hostname:~/curr_path$ sudo reboot
```

C. Check the installed NVIDIA driver version.<br />
```bash
usrname@hostname:~/curr_path$ nvidia-smi
```
```bash
    Mon Dec 24 10:51:12 2018       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.78       Driver Version: 410.78       CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
    | 23%   41C    P8    12W / 250W |    463MiB / 12192MiB |      8%      Default |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0       951      G   /usr/lib/xorg/Xorg                           266MiB |
    |    0      1777      G   compiz                                       148MiB |
    |    0      2047      G   ...uest-channel-token=14304842928010396328    45MiB |
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
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/8_CUDA_toolkit/1.png" width="80%"/><br />

B. Install the CUDA toolkit which user selects.<br />
```bash
usrname@hostname:~/curr_path$ sudo chmod +x cuda_10.0.130_410.48_linux.run
usrname@hostname:~/curr_path$ sudo ./cuda_10.0.130_410.48_linux.run --override
```
```bash
    Do you accept the previously read EULA?
    accept/decline/quit: (accept)
    
    # You are attempting to install on an unsupported configuration. Do you wish to continue?
    # (y)es/(n)o [ default is no ]: (yes)
    
    Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 410.48?
    (y)es/(n)o/(q)uit: (no)
    
    Install the CUDA 10.0 Toolkit?
    (y)es/(n)o/(q)uit: (yes)
    
    Enter Toolkit Location
    [ default is /usr/local/cuda-10.0 ]: (ENTER)
    
    Do you want to install a symbolic link at /usr/local/cuda?  
    (y)es/(n)o/(q)uit: (yes)
    
    Install the CUDA 10.0 Samples?
    (y)es/(n)o/(q)uit: (no)
```

C. Ignore the below warning about incompleted installation.<br /> 
```bash
    ===========
    = Summary =
    ===========
    
    Driver:   Not Selected
    Toolkit:  Installed in /usr/local/cuda-10.0
    Samples:  Not Selected
    
    Please make sure that
     -   PATH includes /usr/local/cuda-10.0/bin
     -   LD_LIBRARY_PATH includes /usr/local/cuda-10.0/lib64, or, add /usr/local/cuda-10.0/lib64 to /etc/ld.so.conf and run ldconfig as root
    
    To uninstall the CUDA Toolkit, run the uninstall script in /usr/local/cuda-10.0/bin
    
    Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.0/doc/pdf for detailed information on setting up CUDA.
    
    ***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 384.00 is required for CUDA 10.0 functionality to work.
    To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run -silent -driver
    
    Logfile is /tmp/cuda_install_5754.log
    Signal caught, cleaning up
```
```bash
usrname@hostname:~/curr_path$ sudo ./cuda_10.0.130_410.48_linux.run -silent -driver
```

<details>
    <summary>D. (Option) Update CUDA toolkits.</summary>
    
    usrname@hostname:~/curr_path$ sudo chmod +x cuda_9.0.176.1_linux.run
    usrname@hostname:~/curr_path$ sudo ./cuda_9.0.176.1_linux.run
    
        NVIDIA CUDA Toolkit
        Do you accept the previously read EULA?
        accept/decline/quit: (accept)
        
        Enter CUDA Toolkit installation directory
        [ default is /usr/local/cuda-9.0 ]: (ENTER)

</details>

E. Make sure that CUDA path and LD_LIBRARY_path.<br />
```bash
usrname@hostname:~/curr_path$ echo -e "\n## CUDA and cuDNN paths"  >> ~/.bashrc
# usrname@hostname:~/curr_path$ echo 'export CUDA_HOME=/usr/local/cuda-10.0' >> ~/.bashrc
# usrname@hostname:~/curr_path$ echo 'export PATH=/usr/local/cuda-10.0/bin' >> ~/.bashrc
# usrname@hostname:~/curr_path$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64' >> ~/.bashrc
usrname@hostname:~/curr_path$ echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
usrname@hostname:~/curr_path$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
usrname@hostname:~/curr_path$ source ~/.bashrc
```
```bash
# usrname@hostname:~/curr_path$ sudo sh -c "echo '/usr/local/cuda-10.0/lib64' >> /etc/ld.so.conf.d/cuda.conf"
# usrname@hostname:~/curr_path$ sudo sh -c "echo '/usr/local/cuda-10.0/lib' >> /etc/ld.so.conf.d/cuda.conf"
# usrname@hostname:~/curr_path$ sudo ldconfig
usrname@hostname:~/curr_path$ sudo reboot
```

F. Check the installed CUDA toolkit version.<br />
```bash
usrname@hostname:~/curr_path$ nvcc --version
```
```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2018 NVIDIA Corporation
    Built on Sat_Aug_25_21:08:01_CDT_2018
    Cuda compilation tools, release 10.0, V10.0.130
```
```bash
usrname@hostname:~/curr_path$ which nvcc
```
```bash
    /usr/local/cuda-10.0/bin/nvcc
```

G. Uninstall the installed CUDA toolkit.<br />
```bash
usrname@hostname:~/curr_path$ sudo ./usr/local/cuda-10.0/bin/uninstall_cuda_10.0.pl
```


## 9. How to install a cuDNN <a name="cudnn"></a>
A. Download a cuDNN (e.g. cuDNN v7.4.2 Library for Linux) with reference to the websites,
<a href="https://developer.nvidia.com/rdp/cudnDownloadn-download" title="cuDNN">
cuDNN
</a>
, 
<a href="https://developer.nvidia.com/rdp/cudnn-archive" title="cuDNN archive">
cuDNN archive
</a>
.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/9_cuDNN/1.png" width="80%"/><br />

B. Install the downloaded cuDNN.<br />
```bash
usrname@hostname:~/curr_path$ tar xzvf cudnn-10.0-linux-x64-v7.4.2.24.tgz
usrname@hostname:~/curr_path$ sudo cp cuda/lib64/* /usr/local/cuda-10.0/lib64/
usrname@hostname:~/curr_path$ sudo cp cuda/include/* /usr/local/cuda-10.0/include/
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h
```

C. Check the installed cuDNN version.<br />
```bash
usrname@hostname:~/curr_path$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
```bash
    #define CUDNN_MAJOR      7
    #define CUDNN_MINOR      4
    #define CUDNN_PATCHLEVEL 2
    --
    #define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
    
    #include "driver_types.h"
```

D. Install the NVIDIA CUDA profiler tools interface.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install libcupti-dev
```


## 10. How to install and use an pip3 and virtualenv <a name="pip3_virtualenv"></a>
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

B. Install the pip3.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install python3-pip
```

C. Check the installed pip3 version.<br />
```bash
usrname@hostname:~/curr_path$ pip3 --version
```
```bash
    pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)
```

D. Install the virtualenv.<br />
```bash
usrname@hostname:~/curr_path$ pip3 install virtualenv
```
```bash
    Installing collected packages: virtualenv
    Successfully installed virtualenv-16.4.3
```

E. Check the installed virtualenv version.<br />
```bash
usrname@hostname:~/curr_path$ virtualenv --version
```
```bash
    16.4.3
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


## 11. How to install and use an Anaconda <a name="conda"></a>
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


## 12. How to install a PyTorch <a name="pytorch"></a>
A. Check a PyTorch version with reference to the website,
<a href="https://pytorch.org" title="PyTorch">
PyTorch
</a>
.<br />

B. Install the PyTorch where user want to install it.<br />
&nbsp; &nbsp; The name of conda virtual environment where user want to install the PyTorch: conda_pytorch<br />
```bash
usrname@hostname:~/curr_path$ conda activate conda_pytorch
(conda_pytorch) usrname@hostname:~/curr_path$ conda install pytorch torchvision cuda100 -c pytorch
```

C. Make sure the PyTorch is installed correctly. <br />
```bash
(conda_pytorch) usrname@hostname:~/curr_path$ python
Python 3.7.1 (default, Dec 14 2018, 19:28:38) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
```
```python
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'TITAN Xp'
>>> torch.cuda.device_count()
1
>>> a = torch.rand(5)
>>> b = a.cuda()
>>> a
tensor([0.4732, 0.1292, 0.7363, 0.6000, 0.2162])
>>> b
tensor([0.4732, 0.1292, 0.7363, 0.6000, 0.2162], device='cuda:0')
```


## 13. How to install a TensorFlow <a name="tensorflow"></a>
A. Check a TensorFlow version with reference to the website,
<a href="https://www.tensorflow.org" title="TensorFlow">
TensorFlow
</a>
.<br />

B. Install the TensorFlow where user want to install it.<br />
&nbsp; &nbsp; The name of conda virtual environment where user want to install the TensorFlow: conda_tf<br />
```bash
usrname@hostname:~/curr_path$ conda activate conda_tf
(conda_tf) usrname@hostname:~/curr_path$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl
```


## 14. How to set an Pycharm environment <a name="pycharm"></a>
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


## 15. Others <a name="others"></a>
A. How to fix NTFS disk write-protect.<br />
```bash
usrname@hostname:~/curr_path$ sudo ntfsfix /dev/sdb1
```

