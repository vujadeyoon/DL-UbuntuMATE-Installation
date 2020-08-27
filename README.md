# Deep Learning Framework Installation on Ubuntu MATE 18.04 LTS
- Deep Learning Framework Installation on Ubuntu MATE 18.04 LTS
- I recommend that you should ignore the commented instructions with an octothorpe, #.
- Modified date: Aug. 27, 2020.

## Table of contents
0.  [Summarized environments about the DL-UbuntuMATE18.04LTS-Installation](#envs)
1.  [How to set an additional language](#language)
2.  [How to remove the Firefox browser and install other browsers](#web_browser)
3.  [How to install a editor, Atom](#Atom)
4.  [How to install a editor, Remarkable](#Remarkable)
5.  [How to install a git](#git)
6.  [How to install a gedit](#gedit)
7.  [How to install a KolourPaint](#KolourPaint)
8.  [How to install a Audacity](#Audacity)
9.  [How to install and set a Remmina](#remmina)
10.  [How to install, uninstall and set both ssh and scp](#ssh_scp)
11.  [How to enable a file system, exfat](#exfat)
12.  [How to install a GPU driver](#gpu_driver)
13.  [How to install a CUDA toolkit](#cuda_toolkit)
14.  [How to install a cuDNN](#cudnn)
15.  [How to install other CUDA toolkit with cuDNN](#cuda_toolkit_cuDNN_other)
16.  [How to uninstall the GPU driver, CUDA toolkit and cuDNN](#uninstall_CUDAs)
17. [How to install python 3.7](#python3.7)
18. [How to install and use pip, pip3 and virtualenv](#pip_virtualenv)
19. [How to install and use an Anaconda](#conda)
20. [How to install a PyTorch](#pytorch)
21. [How to install a TensorFlow](#tensorflow)
22. [How to install both TensorRT and Torch2TRT](#tensorrt_torch2trt)
23. [How to set an Pycharm environment](#pycharm)
24. [Others](#others)


## 0. Summarized environments about the DL-UbuntuMATE18.04LTS-Installation <a name="envs"></a>
- Operating System (OS): Ubuntu MATE 18.04.3 LTS (Bionic)
    - I do not recommend the OS, Ubuntu MATE 18.04.5 LTS (Bionic) because it may be unstable.
- Graphics Processing Unit (GPU): NVIDIA TITAN Xp, 1ea
- GPU driver: Nvidia-440.100
- CUDA toolkit:
    - CUDA-10.2 (default)
    - CUDA-10.1
- cuDNN: cuDNN v7.6.5
- PyTorch: 1.6.0
- TensorFlow (TF): 2.3.0
    - <=CUDA-10.1
    - <= cuDNN v7.6.5


## 1. How to set an additional language <a name="language"></a>
A. Run the Language Support program and install it completely.<br />
B. Select a fcitx option for a keyboard input method system.<br />
&nbsp; &nbsp; Note that the fcitx option is valid when rerunning the program.<br /> 
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/1.png" width="80%"/><br />
C. Logout and login<br />
D. Click the configure tab of a keyboard icon in the upper right corner of the desktop.<br />
E. Add input method (e.g. Hangul).<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/2.png" width="80%"/><br />
F. Set an input method configuration.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/1_Language/3.png" width="80%"/><br />


## 2. How to remove the Firefox browser and install other browsers <a name="web_browser"></a>
A. Reference to the website,
<a href="https://www.opera.com" title="Opera">
Opera
</a>,
<a href="https://www.google.com/chrome">
Google Chrome
</a>.<br />
B. Remove the Firefox browser.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get remove --purge firefox
```
C. Install the Opera and Google Chrome browsers using the package installers after downloading them.<br />
D. Install the Opera addons such as Nimbus Screen Capture and Google Translate.<br />
E. Use the function key F11 when not moving the browsers.<br />


## 3. How to install a editor, Atom <a name="Atom"></a>
A. Reference to the website,
<a href="https://atom.io" title="Atom">
Atom
</a>.<br />
B. Install the Atom using a package installer after downloading it.<br />


## 4. How to install a editor, Remarkable <a name="Remarkable"></a>
A. Reference to the website,
<a href="https://remarkableapp.github.io" title="Remarkable">
Remarkable
</a>.<br />
B. Install the Remarkable using a package installer after downloading it.<br />


## 5. How to install a git <a name="git"></a>
A. Install the gedit.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install git
```


## 6. How to install a gedit <a name="gedit"></a>
A. Install the gedit.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install gedit
```


## 7. How to install a KolourPaint <a name="Kolourpaint"></a>
A. Install the KolourPaint.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install kolourpaint4
```


## 8. How to install a Audacity <a name="Audacity"></a>
A. Install the Audacity.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install audacity
```


## 9. How to install and set a Remmina <a name="remmina"></a>
A. Reference to the website,
<a href="https://remmina.org" title="Remmina">
Remmina
</a>.<br />

B. Install the Remmina.<br />
```bash
usrname@hostname:~/curr_path$ sudo snap install remmina
```

C. Set the Remmina remote desktope preference.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/4_Remmina/1.png" width="80%"/><br />


## 10. How to install, uninstall and set both ssh and scp <a name="ssh_scp"></a>
A. Install the ssh-server.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install openssh-server
```

B. Uninstall the ssh-server.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get remove --purge openssh-server
```


## 11. How to enable a file system, exfat <a name="exfat"></a>
A. Enable the exfat file system.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get install exfat-utils exfat-fuse
```


## 12. How to install a GPU driver <a name="gpu_driver"></a>
A. Check a NVIDIA driver version with reference to the website,
<a href="https://www.nvidia.com/Download/Find.aspx" title="NVIDIA driver">
NVIDIA driver
</a>.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/7_GPU_driver/1.png" width="80%"/><br />

B. Install the NVIDIA driver which user selects.<br />
```bash
usrname@hostname:~/curr_path$ sudo add-apt-repository ppa:graphics-drivers/ppa
usrname@hostname:~/curr_path$ sudo apt-get update
usrname@hostname:~/curr_path$ sudo apt-get install nvidia-driver-440
usrname@hostname:~/curr_path$ sudo reboot
```

C. Check the installed NVIDIA driver version.<br />
```bash
usrname@hostname:~/curr_path$ nvidia-smi
```
```bash
    Sat Aug 22 21:44:52 2020
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.100      Driver Version: 440.100      CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
    | 28%   46C    P8    14W / 250W |    205MiB / 12192MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                               
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      1088      G   /usr/lib/xorg/Xorg                           205MiB |
    +-----------------------------------------------------------------------------+
```


## 13. How to install a CUDA toolkit <a name="cuda_toolkit"></a>
A. Download a CUDA toolkit with reference to the websites,
<a href="https://developer.nvidia.com/cuda-downloads" title="CUDA toolkit">
CUDA toolkit
</a>
and
<a href="https://developer.nvidia.com/cuda-toolkit-archive" title="CUDA toolkit archive">
CUDA toolkit archive
</a>.<br />
&nbsp; &nbsp; Additional reference to the website, 
<a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#abstract" title="NVIDIA CUDA Installation Guide for Linux">
NVIDIA CUDA Installation Guide for Linux
</a>.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/8_CUDA_toolkit/10.1.png" width="80%"/><br />

B. Install the CUDA toolkit which user selects.<br />
```bash
usrname@hostname:~/curr_path$ sudo chmod +x cuda_10.2.89_440.33.01_linux.run
usrname@hostname:~/curr_path$ sudo ./cuda_10.2.89_440.33.01_linux.run --override
```
```bash
    # Ignore the below warning and just select a option, Continue.
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ Existing package manager installation of the driver found. It is strongly    │
    │ recommended that you remove this before continuing.                          │
    │ Abort                                                                        │
    │ (Continue)                                                                   │
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
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │ Up/Down: Move | 'Enter': Select                                              │
    └──────────────────────────────────────────────────────────────────────────────┘

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
    │      [ ] 440.33.01                                                           │
    │ - [X] CUDA Toolkit 10.2                                                      │
    │    + [X] CUDA Tools 10.2                                                     │
    │    + [X] CUDA Libraries 10.2                                                 │
    │    + [X] CUDA Compiler 10.2                                                  │
    │      [X] CUDA Misc Headers 10.2                                              │
    │   [ ] CUDA Samples 10.2                                                      │
    │   [ ] CUDA Demo Suite 10.2                                                   │
    │   [ ] CUDA Documentation 10.2                                                │
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
```

C. Ignore the below warning about incompleted installation.<br /> 
```bash
    ===========
    = Summary =
    ===========

    Driver:   Not Selected
    Toolkit:  Installed in /usr/local/cuda-10.2/
    Samples:  Not Selected

    Please make sure that
     -   PATH includes /usr/local/cuda-10.2/bin
     -   LD_LIBRARY_PATH includes /usr/local/cuda-10.2/lib64, or, add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root

    To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-10.2/bin

    Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.2/doc/pdf for detailed information on setting up CUDA.
    ***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 440.00 is required for CUDA 10.2 functionality to work.
    To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
        sudo <CudaInstaller>.run --silent --driver

    Logfile is /var/log/cuda-installer.log
```

D. Make sure that CUDA path and LD_LIBRARY_PATH.<br />
```bash
usrname@hostname:~/curr_path$ echo -e "\n## CUDA and cuDNN paths"  >> ~/.bashrc
usrname@hostname:~/curr_path$ echo 'export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}' >> ~/.bashrc
usrname@hostname:~/curr_path$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
usrname@hostname:~/curr_path$ source ~/.bashrc
usrname@hostname:~/curr_path$ sudo reboot
```

E. Check the installed CUDA toolkit version.<br />
```bash
usrname@hostname:~/curr_path$ nvcc --version
```
```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_19:24:38_PDT_2019
    Cuda compilation tools, release 10.2, V10.2.89
```
```bash
usrname@hostname:~/curr_path$ which nvcc
```
```bash
    /usr/local/cuda-10.2/bin/nvcc
```


## 14. How to install a cuDNN <a name="cudnn"></a>
A. Download a cuDNN (e.g. cuDNN v7.6.5 Library for Linux) with reference to the websites,
<a href="https://developer.nvidia.com/rdp/cudnn-download" title="cuDNN">
cuDNN
</a>, 
<a href="https://developer.nvidia.com/rdp/cudnn-archive" title="cuDNN archive">
cuDNN archive
</a>.<br />
<img src="https://github.com/vujadeyoon/DL-UbuntuMATE18.04LTS-Installation/blob/master/Figures/9_cuDNN/7.6.5.png" width="80%"/><br />

B. Install the downloaded cuDNN.<br />
```bash
usrname@hostname:~/curr_path$ tar xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgz
usrname@hostname:~/curr_path$ sudo cp cuda/lib64/* /usr/local/cuda-10.2/lib64/
usrname@hostname:~/curr_path$ sudo cp cuda/include/* /usr/local/cuda-10.2/include/
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.2/lib64/libcudnn*
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.2/include/cudnn.h
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


## 15. How to install other CUDA toolkit with cuDNN <a name="cuda_toolkit_cuDNN_other"></a>
A. This is for cases where you need to use a different CUDA toolkit (e.g. cuda-10.1) with cuDNN (e.g. cudnn-10.1-linux-x64-v7.6.5.32.tgz).<br />
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
    
    # Select a option, No.
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ A symlink already exists at /usr/local/cuda. Update to this installation?    │
    │ Yes                                                                          │
    │ (No)                                                                         │
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
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │                                                                              │
    │ Up/Down: Move | 'Enter': Select                                              │
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

D. Install the cuDNN which user selects.<br />
```bash
usrname@hostname:~/curr_path$ tar xzvf cudnn-10.1-linux-x64-v7.6.5.32.tgz
usrname@hostname:~/curr_path$ sudo cp cuda/lib64/* /usr/local/cuda-10.1/lib64/
usrname@hostname:~/curr_path$ sudo cp cuda/include/* /usr/local/cuda-10.1/include/
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*
usrname@hostname:~/curr_path$ sudo chmod a+r /usr/local/cuda-10.1/include/cudnn.h
```

E. Check the installed cuDNN version.<br />
```bash
usrname@hostname:~/curr_path$ cat /usr/local/cuda-10.1/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
```bash
    #define CUDNN_MAJOR 7
    #define CUDNN_MINOR 6
    #define CUDNN_PATCHLEVEL 5
    --
    #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

    #include "driver_types.h"
```


## 16. How to uninstall the GPU driver, CUDA toolkit and cuDNN <a name="uninstall_CUDAs"></a>
A. uninstall the GPU driver, CUDA toolkit (e.g. cuda-10.2) and cuDNN.<br />
```bash
usrname@hostname:~/curr_path$ sudo /usr/local/cuda-10.2/bin/cuda-uninstaller
usrname@hostname:~/curr_path$ sudo apt-get remove --purge -y 'cuda*'
usrname@hostname:~/curr_path$ sudo apt-get remove --purge -y 'nvidia*'
usrname@hostname:~/curr_path$ sudo apt-get autoremove --purge cuda
usrname@hostname:~/curr_path$ sudo rm -rf /usr/local/cuda*
```


## 17. How to install python 3.7 <a name="python3.7"></a>
A. Install the python3.7.<br />
```bash
usrname@hostname:~/curr_path$ sudo apt-get update
usrname@hostname:~/curr_path$ sudo apt-get install software-properties-common
usrname@hostname:~/curr_path$ sudo add-apt-repository ppa:deadsnakes/ppa
usrname@hostname:~/curr_path$ (ENTER)
usrname@hostname:~/curr_path$ sudo apt-get install python3.7
```

B. Check the installed python3.7 version.<br />
```bash
usrname@hostname:~/curr_path$ python3.7 --version
```
```bash
    Python 3.7.9
```
```bash
usrname@hostname:~/curr_path$ python3.7
```
```bash
    Python 3.7.9 (default, Aug 18 2020, 06:22:45) 
    [GCC 7.5.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.

```


## 18. How to install and use pip, pip3 and virtualenv <a name="pip_virtualenv"></a>
A. Check the pip, pip3 and virtualenv usages with reference to the websites,
<a href="https://pip.pypa.io/en/stable/" title="Pip3">
pip3
</a>,
<a href="https://virtualenv.pypa.io/en/latest/" title="Virtualenv1">
virtualenv1
</a>
and
<a href="https://packaging.python.org/guides/installing-using-pip-and-virtualenv/" title="Virtualenv2">
virtualenv2
</a>.<br />

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
# You must install the virtualenv as root whose version is 16.7.9 using the pip, not the pip3. Other versions may be unstable.
usrname@hostname:~/curr_path$ sudo pip install virtualenv==16.7.9
```
```bash
    Installing collected packages: virtualenv
    Successfully installed virtualenv-16.7.9
```

E. Check the installed version.<br />
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


## 19. How to install and use an Anaconda <a name="conda"></a>
A. Download an Anaconda with reference to the website,
<a href="https://www.anaconda.com/download/#linux" title="Anaconda">
Anaconda
</a>.<br />

B. Install the downloaded Anaconda.<br />
```bash
usrname@hostname:~/curr_path$ bash Anaconda3-2019.10-Linux-x86_64.sh
```
```bash
    Do you accept the license terms? [yes|no]
    [no] >>> (yes)
    Anaconda3 will now be installed into this location:
    /home/usrname/anaconda3
    [/home/usrname/anaconda3] >>> (ENTER)
    Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]
    [no] >>> (no)
```
```bash
usrname@hostname:~/curr_path$ source ~/.bashrc
```
* Please note that the official Anaconda installation guide recommends that the installer initialize the Anaconda3 by running conda init while installing it. However, I recommend that you type no for the last question in installation process, because the command, conda init, modifies the file, ~/.bashrc, then a terminal always activates a base conda virtual envrionment as below.
```bash
(base) usrname@hostname:~/curr_path$
```
* It enables the terminal to call any conda command easliy, but the automatically activated conda virtual envrionment, base, is redundant if you use the virtualenv that I mentioned in the section 11.
* Thus, after installing the Anaconda3 without initializing the conda, you should activate the conda virtual environment, base to use conda virtual environment correctly as below.
```bash
usrname@hostname:~/curr_path$ source /home/usrname/anaconda3/bin/activate
(base) usrname@hostname:~/curr_path$ 
```
* If you want to initialize the conda after installing it without initializing process, you run a command, conda init, after activating the base conda virtual environment. It can modify the file, ~/.bashrc. Please refer to below information. 
```bash
(base) usrname@hostname:~/curr_path$ conda init
```
```bash
(base) usrname@hostname:~/curr_path$ cat ~/.bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/usrname/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/usrname/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/usrname/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/usrname/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
<details>
    <summary>C. (Option) Update the Anaconda.</summary>
    
    (base) usrname@hostname:~/curr_path$ conda update conda
    
        The following packages will be UPDATED:
        Proceed ([y]/n)? (y)
        
</details>


D. Uninstall the Anaconda.<br />
```bash
usrname@hostname:~/curr_path$ rm -rf ~/anaconda3
```

E. Check the installed conda version.<br />
```bash
(base) usrname@hostname:~/curr_path$ conda --version
```
```bash
    conda 4.7.12
```

F. Check the conda envrionments.<br />
```bash
(base) usrname@hostname:~/curr_path$ conda info --envs
```
```bash
    # conda environments:
    #
    base                  *  /home/usrname/anaconda3
    
```

G. Create a conda virtual environments for python 3.7 with conda.<br />
&nbsp; &nbsp; The name of new conda virtual environment to be created: conda_dl<br />
```
(base) usrname@hostname:~/curr_path$ conda create --name conda_dl python=3.7
```

H. Clone a conda virtual environment.<br />
&nbsp; &nbsp; The name of new conda virtual environment to be cloned: conda_pytorch<br />
&nbsp; &nbsp; The name of existed conda virtual environment: conda_dl<br />
```bash
(base) usrname@hostname:~/curr_path$ conda create --name conda_pytorch --clone conda_dl
```

I. Activate a conda virtual environment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be activated: conda_pytorch<br />
```bash
(base) usrname@hostname:~/curr_path$ conda activate conda_pytorch
```

J. Deactivate a conda virtual environment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be deactivated: conda_pytorch<br />
```bash
(conda_pytorch) usrname@hostname:~/curr_path$ conda deactivate
```

K. Remove a conda virtual environment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be removed: conda_dl<br />
```bash
(base) usrname@hostname:~/curr_path$ conda remove --name conda_dl --all
```

L. Export a conda package list.<br />
&nbsp; &nbsp; The name of activated conda virtual environment: conda_pytorch<br />
```bash
(conda_pytorch) usrname@hostname:~/curr_path$ conda list -e > requirements.txt
```

M. Install packages from the exported conda package list.<br />
&nbsp; &nbsp; The name of activated conda virtual environment: conda_new<br />
```bash
(conda_new) usrname@hostname:~/curr_path$ conda install --yes --file requirements.txt # does not automatically install all the dependencies
```
```bash
(conda_new) usrname@hostname:~/curr_path$ while read requirement; do conda install --yes $requirement; done < requirements.txt # automatically install all the dependencies
```

N. Export a conda virtual envrionment.<br />
&nbsp; &nbsp; The name of conda virtual environment to be exported: conda_pytorch<br />
&nbsp; &nbsp; The name of exported file: exported_env.yml<br />
```bash
(base) usrname@hostname:~/curr_path$ conda conda_pytorch export > exported_env.yml
```

O. Create a conda virtual environment with the exported conda virtual environment.<br />
&nbsp; &nbsp; The name of new conda virtual environment to be created: conda_new<br />
&nbsp; &nbsp; The name of exported file: exported_env.yml<br />
```bash
(base) usrname@hostname:~/curr_path$ conda conda_new create -f exported_env.yml
```


## 20. How to install a PyTorch <a name="pytorch"></a>
A. Check a PyTorch version with reference to the website,
<a href="https://pytorch.org" title="PyTorch">
PyTorch
</a>.<br />

B. Install the PyTorch where user want to install it.<br />
&nbsp; &nbsp; The name of virtualenv where user want to install the PyTorch: virenv_pytorch<br />
```bash
usrname@hostname:~/curr_path$ source /home/usrname/pip3_virtualenv/virenv_pytorch/bin/activate
(virenv_pytorch) usrname@hostname:~/curr_path$ pip3 install torch torchvision
```

C. Install other version of the PyTorch for older CUDA toolkit.<br />
&nbsp; &nbsp; The name of virtualenv where user want to install the PyTorch: virenv_pytorch<br />
&nbsp; &nbsp; The PyTorch version: 1.4.0<br />
&nbsp; &nbsp; The Torchvision version: 0.5.0<br />
&nbsp; &nbsp; The CUDA toolkit version: 10.0<br />
```bash
usrname@hostname:~/curr_path$ source /home/usrname/pip3_virtualenv/virenv_pytorch/bin/activate
(virenv_pytorch) usrname@hostname:~/curr_path$ pip3 install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

D. Make sure the PyTorch is installed correctly. <br />
```bash
(virenv_pytorch) usrname@hostname:~/curr_path$ python3
Python 3.7.9 (default, Aug 18 2020, 06:22:45)
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```
```python
>>> import torch
>>> torch.__version__
'1.6.0'
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name(0)
'TITAN Xp'
>>> device = torch.device('cuda')
>>> print(str(device))
'cuda'
>>> a = torch.rand(5)
>>> b = a.to(device)
>>> c = a.to('cuda:0')
>>> print(a)
tensor([0.4732, 0.1292, 0.7363, 0.6000, 0.2162])
>>> print(b)
tensor([0.4732, 0.1292, 0.7363, 0.6000, 0.2162], device='cuda:0')
>>> print(c)
tensor([0.4732, 0.1292, 0.7363, 0.6000, 0.2162], device='cuda:0')
```


## 21. How to install a TensorFlow <a name="tensorflow"></a>
A. Check a TensorFlow version with reference to the website,
<a href="https://www.tensorflow.org" title="TensorFlow">
TensorFlow
</a>.<br />

B. Register environmental variables temporarily (e.g. CUDA path and LD_LIBRARY_PATH) because the TF2.3 supports CUDA toolkit 10.1, not CUDA toolkit 10.2.<br />
```bash
usrname@hostname:~/curr_path$ export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
usrname@hostname:~/curr_path$ export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

C. Install the TensorFlow where user want to install it.<br />
&nbsp; &nbsp; The name of virtualenv where user want to install the TensorFlow: virenv_tf<br />
```bash
usrname@hostname:~/curr_path$ source /home/usrname/pip3_virtualenv/virenv_tf/bin/activate
(virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow
(virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow-gpu
# The final version of TensorFlow 1.x:
# (virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow==1.15
# (virenv_tf) usrname@hostname:~/curr_path$ pip3 install tensorflow-gpu==1.15
```

D. Make sure the TensorFlow is installed correctly. <br />
```bash
(virenv_tf) usrname@hostname:~/curr_path$ python3
Python 3.7.9 (default, Aug 18 2020, 06:22:45)
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```
```python
>>> import tensorflow as tf
2020-08-22 23:17:27.798905: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
>>> tf.__version__
'2.3.0'
>>> tf.test.is_gpu_available()
WARNING:tensorflow:From <stdin>:1: is_gpu_available ~ GPU (device: 0, name: TITAN Xp, pci bus id: 0000:01:00.0, compute capability: 6.1)
True
>>> tf.debugging.set_log_device_placement(True)
>>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
>>> c = tf.matmul(a, b)
2020-08-22 23:18:26.180782: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-08-22 23:18:26.878503: I tensorflow/core/common_runtime/eager/execute.cc:611] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
>>> print(c)
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```

E. Unregister the registered temporal environmental variables (e.g. CUDA path and LD_LIBRARY_PATH).<br />
```bash
usrname@hostname:~/curr_path$ unset PATH
usrname@hostname:~/curr_path$ unset LD_LIBRARY_PATH
```


## 22. How to install both TensorRT and Torch2TRT <a name="tensorrt_torch2trt"></a>
A. Reference to the website,
<a href="https://github.com/vujadeyoon/TensorRT-Torch2TRT" title="TensorRT-Torch2TRT">TensorRT-Torch2TRT</a>.<br />


## 23. How to set a Pycharm environment <a name="pycharm"></a>
A. Download a Pycharm which is a kind of Python IDEs with reference to the website,
<a href="https://www.jetbrains.com/pycharm/download/#section=linux" title="Pycharm">
Pycharm
</a>.<br />

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


## 24. Others <a name="others"></a>
A. How to fix NTFS disk write-protect.<br />
```bash
usrname@hostname:~/curr_path$ sudo ntfsfix /dev/sdb1
```
B. How to install citrix.<br />
- Download and install the citrix recevier (e.g. Citrix Receiver 13.10 for Linux) for Debian Packages (i.e. Full Packages (Self-Service Support)).
- Then copy some certification files to the Citrix directory.
```bash
usrname@hostname:~/curr_path$ sudo cp -r /etc/ssl/certs/* /opt/Citrix/ICAClient/keystore/cacerts/
```
