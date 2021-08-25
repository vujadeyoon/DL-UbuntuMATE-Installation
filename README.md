# Deep Learning Framework Installation on Ubuntu MATE 20.04 LTS
* Deep Learning Framework Installation on Ubuntu MATE 20.04 LTS
* I recommend that you should ignore the commented instructions with an octothorpe, #.
* I recommend that you should type the sercert strings with an barakets, <>.
* Modified date: Aug. 26, 2021.


## Table of contents
1.  [Summarized environments about the DL-UbuntuMATE-Installation](#envs)
2.  [How to set an additional language](#language)
3.  [How to remove unused packages](#remove_packages)
4.  [How to install useful packages](#install_packages)
5.  [How to install a GPU driver](#gpu_driver)
6.  [How to install a CUDA toolkit](#cuda_toolkit)
7.  [How to install a cuDNN](#cudnn)
8.  [How to uninstall the GPU driver, CUDA toolkit and cuDNN](#uninstall_CUDAs)
9.  [How to install other version of the python3](#python3_version)
10. [How to install and use pip3 and venv](#pip3_venv)
11. [How to install a PyTorch](#pytorch)
12. [How to install both TensorRT and Torch2TRT](#tensorrt_torch2trt)
13. [How to set an Pycharm environment](#pycharm)
14. [Development tools](#dev_tools)
15. [Citrix](#citrix)
16. [File system](#file_system)
17. [File mode](#file_mode)


## 1. Summarized environments about the DL-UbuntuMATE-Installation <a name="envs"></a>
* Operating System (OS): Ubuntu MATE 20.04.2 LTS (Focal)
* Graphics Processing Unit (GPU): NVIDIA TITAN Xp, 1ea
* GPU driver: Nvidia-460.91.03
* CUDA toolkit:
    * CUDA-11.1
* cuDNN: cuDNN v8.2.1
* PyTorch: 1.9.0


## 2. How to set an additional language <a name="language"></a>
A. Run the Language Support program and install it completely.

B. Install the debian package, IBus.
```bash
usrname@hostname:~/path_curr$ sudo apt-get install ibus
```

C. Select the IBus option for a keyboard input method system.
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/Language/1.png" width="80%"/>

D. Logout and login the OS.

E. Run the ibus-setup.
```bash
usrname@hostname:~/path_curr$ ibus-setup
```

F. Add the Korean - Hangul at the Input Method tap.
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/Language/2.png" width="80%"/>

G. Logout and login the OS.

H. Set the language at the upper right corner of the desktop.


## 3. How to remove unused packages <a name="remove_packages"></a>
### Debian packages
A. Firfox and openssh-server.
```bash
usrname@hostname:~/path_curr$ sudo apt-get remove --purge firefox openssh-server
```


## 4. How to install useful packages <a name="install_packages"></a>
### Debian packages
A. Web browser: <a href="https://www.opera.com" title="Opera"> Opera</a> and <a href="https://www.google.com/chrome" title="Google Chrome"> Google Chrome</a>.
* Install the Opera and Google Chrome browsers using the package installers after downloading them.
* Install the Opera addons.
    * Nimbus Screen Capture
    * Google Translate
* Enable the Block ads and surf the web up to three times faster and Block trackers in the Settings.

B. Simple Screen Recoder
```bash
usrname@hostname:~/path_curr$ sudo add-apt-repository ppa:maarten-baert/simplescreenrecorder
usrname@hostname:~/path_curr$ sudo apt-get update
usrname@hostname:~/path_curr$ sudo apt-get install simplescreenrecorder
```

C. Others: curl, terminator, gedit, kolourpaint4, audacity, filezilla and openssh-server.
```bash
usrname@hostname:~/path_curr$ sudo apt-get install curl terminator git gedit kolourpaint4 audacity filezilla openssh-server
```

### Snap packages
A. Remote desktop: <a href="https://remmina.org" title="Remmina"> Remmina</a>.
* Instsall the Remmina.
* Set the Remmina remote desktope preference.
```bash
usrname@hostname:~/path_curr$ sudo snap install remmina
```
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/Remmina/1.png" width="80%"/>


## 5. How to install a GPU driver <a name="gpu_driver"></a>
A. Check a NVIDIA driver version with reference to the website, <a href="https://www.nvidia.com/Download/Find.aspx" title="NVIDIA driver"> NVIDIA driver</a>.
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/GPU_driver/1.png" width="80%"/>

B. Install the NVIDIA driver which user selects.
```bash
usrname@hostname:~/path_curr$ sudo add-apt-repository ppa:graphics-drivers/ppa
usrname@hostname:~/path_curr$ sudo apt-get update
usrname@hostname:~/path_curr$ sudo apt-get install nvidia-driver-460
usrname@hostname:~/path_curr$ sudo reboot
```

C. Check the installed NVIDIA driver version.
```bash
usrname@hostname:~/path_curr$ nvidia-smi
```
```bash
    Tue Aug 24 00:51:47 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
    | 23%   36C    P8    13W / 250W |    467MiB / 12192MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A       923      G   /usr/lib/xorg/Xorg                304MiB |
    |    0   N/A  N/A      1795      G   ...AAAAAAAAA= --shared-files       79MiB |
    |    0   N/A  N/A     34804      G   ...AAAAAAAAA= --shared-files       80MiB |
    +-----------------------------------------------------------------------------+

```


## 6. How to install a CUDA toolkit <a name="cuda_toolkit"></a>
A. Download a CUDA toolkit with reference to the websites, <a href="https://developer.nvidia.com/cuda-downloads" title="CUDA toolkit"> CUDA toolkit</a> and <a href="https://developer.nvidia.com/cuda-toolkit-archive" title="CUDA toolkit archive"> CUDA toolkit archive</a>.
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/CUDA_toolkit/1.png" width="80%"/>

B. Install the CUDA toolkit which user selects.
* Please note that you can download the CUDA toolkit installation file again if you face below errors with reference to the websites, <a href="https://forums.developer.nvidia.com/t/cuda-installation-error-extraction-failed/50845/5" title="NVIDIA Developer"> NVIDIA Developer</a>.
```bash
    gzip: stdin: invalid compressed data--format violated
    Extraction failed.
    Ensure there is enough space in /tmp and that the installation package is not corrupt
    Signal caught, cleaning up
```
```bash
usrname@hostname:~/path_curr$ wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
usrname@hostname:~/path_curr$ sudo sh cuda_11.1.0_455.23.05_linux.run
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
    │  NVIDIA Software License Agreement and CUDA Supplement to                    │
    │  Software License Agreement.                                                 │
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
    │──────────────────────────────────────────────────────────────────────────────│
    │ Do you accept the above EULA? (accept/decline/quit):                         │
    │ (accept)                                                                     │
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ CUDA Installer                                                               │
    │ - [ ] Driver                                                                 │
    │      [ ] 455.23.05                                                           │
    │ - [X] CUDA Toolkit 11.1                                                      │
    │    + [X] CUDA Tools 11.1                                                     │
    │    + [X] CUDA Compiler 11.1                                                  │
    │    + [X] CUDA Libraries 11.1                                                 │
    │   [ ] CUDA Samples 11.1                                                      │
    │   [ ] CUDA Demo Suite 11.1                                                   │
    │   [ ] CUDA Documentation 11.1                                                │
    │   Options                                                                    │
    │   (Install)                                                                  │
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

C. Ignore the below warning about incompleted installation.
```bash
    ===========
    = Summary =
    ===========

    Driver:   Not Selected
    Toolkit:  Installed in /usr/local/cuda-11.1/
    Samples:  Not Selected

    Please make sure that
    -   PATH includes /usr/local/cuda-11.1/bin
    -   LD_LIBRARY_PATH includes /usr/local/cuda-11.1/lib64, or, add /usr/local/cuda-11.1/lib64 to /etc/ld.so.conf and run ldconfig as root

    To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.1/bin
    ***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least .00 is required for CUDA 11.1 functionality to work.
    To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
        sudo <CudaInstaller>.run --silent --driver

    Logfile is /var/log/cuda-installer.log
```

D. Make sure that CUDA path and LD_LIBRARY_PATH.
```bash
usrname@hostname:~/path_curr$ echo -e "\n## CUDA and cuDNN paths"  >> ~/.bashrc
usrname@hostname:~/path_curr$ echo 'export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
usrname@hostname:~/path_curr$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
usrname@hostname:~/path_curr$ source ~/.bashrc
usrname@hostname:~/path_curr$ sudo reboot
```

E. Check the installed CUDA toolkit version.
```bash
usrname@hostname:~/path_curr$ nvcc --version
```
```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2020 NVIDIA Corporation
    Built on Tue_Sep_15_19:10:02_PDT_2020
    Cuda compilation tools, release 11.1, V11.1.74
    Build cuda_11.1.TC455_06.29069683_0
```
```bash
usrname@hostname:~/path_curr$ which nvcc
```
```bash
    /usr/local/cuda-11.1/bin/nvcc
```

F. Make a symbolic link when changing another installed CUDA toolkit.
```bash
usrname@hostname:~/path_curr$ sudo rm -rf /usr/local/cuda
usrname@hostname:~/path_curr$ sudo ln -s /usr/local/cuda-11.1 /usr/local/cuda
usrname@hostname:~/path_curr$ sudo readlink -f /usr/local/cuda
```
```bash
    /usr/local/cuda-11.1
```


## 7. How to install a cuDNN <a name="cudnn"></a>
A. Download a cuDNN (e.g. cuDNN v7.6.5 Library for Linux) with reference to the websites, <a href="https://developer.nvidia.com/rdp/cudnn-download" title="cuDNN"> cuDNN</a> and <a href="https://developer.nvidia.com/rdp/cudnn-archive" title="cuDNN archive"> cuDNN archive</a>.
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/cuDNN/1.png" width="80%"/>

B. Install the downloaded cuDNN.
```bash
usrname@hostname:~/path_curr$ tar -xzvf cudnn-11.3-linux-x64-v8.2.1.32.tgz 
usrname@hostname:~/path_curr$ sudo cp -r cuda/lib64/* /usr/local/cuda-11.1/lib64/
usrname@hostname:~/path_curr$ sudo cp -r cuda/include/* /usr/local/cuda-11.1/include/
usrname@hostname:~/path_curr$ sudo chmod a+r /usr/local/cuda-11.1/lib64/libcudnn*
usrname@hostname:~/path_curr$ sudo chmod a+r /usr/local/cuda-11.1/include/cudnn.h
```

C. Check the installed cuDNN version.
```bash
usrname@hostname:~/path_curr$ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
```bash
    #define CUDNN_MAJOR 8
    #define CUDNN_MINOR 2
    #define CUDNN_PATCHLEVEL 1
    --
    #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

    #endif /* CUDNN_VERSION_H */
```

D. Install the NVIDIA CUDA profiler tools interface.
```bash
usrname@hostname:~/path_curr$ sudo apt-get install libcupti-dev
```


## 8. How to uninstall the GPU driver, CUDA toolkit and cuDNN <a name="uninstall_CUDAs"></a>
A. uninstall the GPU driver, CUDA toolkit (e.g. cuda-11.1) and cuDNN.
```bash
usrname@hostname:~/path_curr$ sudo /usr/local/cuda-11.1/bin/cuda-uninstaller
usrname@hostname:~/path_curr$ sudo apt-get remove --purge -y 'cuda*'
usrname@hostname:~/path_curr$ sudo apt-get remove --purge -y 'nvidia*'
usrname@hostname:~/path_curr$ sudo apt-get autoremove --purge cuda
usrname@hostname:~/path_curr$ sudo rm -rf /usr/local/cuda*
```


## 9. How to install other version of the python3 <a name="python3_version"></a>
A. Install the other version of the python3: python3.7 (e.g. python3.7.9).
* Installed (default): python3.8
* To be installed: python3.7
```bash
usrname@hostname:~/path_curr$ sudo apt-get update
usrname@hostname:~/path_curr$ sudo apt-get install software-properties-common
usrname@hostname:~/path_curr$ sudo add-apt-repository ppa:deadsnakes/ppa
usrname@hostname:~/path_curr$ sudo apt-get install python3.7
```

B. Make a symbolic link when changing the version of the python3.
* Default: python3.8
* To be changed: python3.7
```bash
usrname@hostname:~/path_curr$ sudo rm -rf /usr/bin/python3
usrname@hostname:~/path_curr$ sudo ln -s /usr/bin/python3.7 /usr/bin/python3
usrname@hostname:~/path_curr$ sudo readlink -f /usr/bin/python3
```
```bash
    /usr/bin/python3.7
```


## 10. How to install and use pip3 and venv <a name="pip3_venv"></a>
A. Check the pip, pip3 and virtualenv usages with reference to the websites, <a href="https://pip.pypa.io/en/stable/" title="pip3"> pip3</a> and <a href="https://docs.python.org/3/library/venv.html#module-venv" title="venv"> venv</a>.

B. Install the pip3.
```bash
usrname@hostname:~/path_curr$ sudo apt-get install python3-pip
```

C. Check the installed pip3 version.
```bash
usrname@hostname:~/path_curr$ pip3 --version
```
```bash
    pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)
```

D. Install the venv.
```bash
usrname@hostname:~/path_curr$ sudo apt-get install python3-venv
```

E. Create a venv for python 3.8 with pip3.
* Please note that set the default python3 version to be installed with the venv refer to the section 9-b before creating the venv.
```bash
usrname@hostname:~/path_curr$ mkdir -p /DATA/python3_venv/
usrname@hostname:~/path_curr$ python3 -m venv /DATA/python3_venv/python3.8_pytorch1.9
```

F. Activate the venv.
```bash
usrname@hostname:~/path_curr$ source /DATA/python3_venv/python3.8_pytorch1.9/bin/activate
```

G. Deactivate the venv.
```bash
(python3.8_pytorch1.9) usrname@hostname:~/path_curr$ deactivate
```

H. Remove the venv.
```bash
usrname@hostname:~/path_curr$ rm -rf /DATA/python3_venv/python3.8_pytorch1.9/
```

I. Export a pip3 package list.
```bash
(python3.8_pytorch1.9) usrname@hostname:~/path_curr$ pip3 freeze > requirements.txt
```

J. Install packages from the exported pip3 package list.
```bash
(python3.8_pytorch1.9) usrname@hostname:~/path_curr$ pip3 install -r requirements.txt
```


## 11. How to install a PyTorch <a name="pytorch"></a>
A. Check a PyTorch version with reference to the website, <a href="https://pytorch.org" title="PyTorch"> PyTorch</a>.

B. Install the PyTorch where user want to install it.
```bash
usrname@hostname:~/path_curr$ pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

C. Make sure the PyTorch is installed correctly.
```bash
usrname@hostname:~/path_curr$ source /DATA/python3_venv/python3.8_pytorch1.9/bin/activate
(python3.8_pytorch1.9) usrname@hostname:~/path_curr$ python3
```
```python
    Python 3.8.10 (default, Jun  2 2021, 10:49:15) 
    [GCC 9.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
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


## 12. How to install both TensorRT and Torch2TRT <a name="tensorrt_torch2trt"></a>
A. Reference to the website, <a href="https://github.com/vujadeyoon/TensorRT-Torch2TRT" title="TensorRT-Torch2TRT">TensorRT-Torch2TRT</a>.


## 13. How to set a Pycharm environment <a name="pycharm"></a>
A. Download a Pycharm which is a kind of Python IDEs with reference to the website, <a href="https://www.jetbrains.com/pycharm/download/#section=linux" title="Pycharm"> Pycharm</a>.

B. Install the Pycharm.
```bash
usrname@hostname:~/path_curr$ tar -xzvf pycharm-community-2021.2.tar.gz
```

C. Run the Pycharm.
```bash
usrname@hostname:~/path_curr$ bash pycharm-community-2021.2/bin/pycharm.sh
```

D. Create a new project with existing interpreter which is in a specific conda environment (e.g. conda_dl).
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/Pycharm/1.png" width="80%"/>

E. How to set a project interpreter.
<br /> <img src="https://github.com/vujadeyoon/DL-UbuntuMATE-Installation/blob/Ubuntu_MATE_20.04_LTS/Figures/Pycharm/2.png" width="80%"/>


## 14. Development tools <a name="dev_tools"></a>
### Git
A. How to set name and eamil globally.
```bash
usrname@hostname:~/path_curr$ git config --global user.name "user_name"
usrname@hostname:~/path_curr$ git config --global user.email "user_email_address"
```

B. How to ignore a notice for difference of the file mode.
```bash
    old mode 100755
    new mode 100644
```
```bash
usrname@hostname:~/path_curr$ git config core.filemode false
```

### Vim
A. How to set numbers globally.
```bash
usrname@hostname:~/path_curr$ vi ~/.vimrc
set number
:wq
```

### Visual Studio Code
A. How to open the Settings.
```bash
Ctrl + ,
```

B. Useful extensions.
* Markdown Preview Enhanced
* Diff Folders

C. How to set live preview.
* Open the Settings and search the string, Auto Save.
* Set the variables.
    * Files: Auto Save: afterDelay
    * Files: Auto Save Delay: 500
* Duplicate the window
```bash
F1 + Workspaces: Duplicate As workspace in New Window
```

D. How to edit font ligatures.
* Open the settings.json for the font ligatures.
    * Open the Settings.
    * Search the settings.
    * Click the button, Edit in the settings.json in the section, Editor: Font Ligatrues.
* Replace the json file.
```
{
    "workbench.colorTheme": "Quiet Light",
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "editor.suggestSelection": "first",
    "vsintellicode.modify.editor.suggestSelection": "automaticallyOverrodeDefaultValue",
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 500,
    "[json]": {
        "editor.quickSuggestions": {
            "strings": true
        },
        "editor.suggest.insertMode": "replace"
    },

    // Custom
    // Minimap
    "editor.minimap.enabled": false,
    // Color
    "workbench.colorCustomizations": {
        // Totally Transparent
        "editor.findMatchHighlightBackground": "#ffff0050",
        // Borders
        "editor.selectionHighlightBorder": "#ff0000",
        "editor.lineHighlightBorder": "#00000020",
        // Selection
        "editor.findMatchBackground": "#ffff00",
        "editor.findMatchBorder": "#ff0000",
        // Misc
        "editorCursor.foreground": "#000000",
        "editor.lineHighlightBackground": "#E3F5D3",

    },
    "window.zoomLevel": -1,
    "editor.fontLigatures": null
}
```

### bashrc
A. How to edit bashrc.
```bash
usrname@hostname:~/path_curr$ vi ~/.bashrc
```

B. How to apply bashrc.
```bash
usrname@hostname:~/path_curr$ source ~/.bashrc
```

C. Useful functions.
* vid2frm.
```bash
## vid2frm: Video to frames.
function vid2frm() {
  case $1 in
    -h|--help)
    echo "Usage: vid2frm name_video.mp4"
    ;;
    *)
    path_file=$1
    filename_extension="${path_file##*/}" # "$(basename -- $path_file)"
    filename=${filename_extension%.*}
    extension=".${path_file##*.}"
    rm -rf ./$filename
    mkdir -p ./$filename
    ffmpeg -i $1 -start_number 0 ./$filename/frm_%08d.png
    ;;
  esac
}
```
* git_init.
```bash
## Git init.
function git_init() {
    number_commit=$1
    rm -rf .git
    git init
    git add .
    git commit -m "git init: ${number_commit}"
}
```
* github_token.
```bash
## Get GitHub token.
function github_token() {
    echo "GitHub token: <github_token>"
}
```
* csv2ods.
```bash
## Convert CSV to ODS.
function csv2ods() {
    soffice --convert-to ods $1
}
```
* docker_connect.
```bash
## docker_connect
function docker_connect() {
    # $1: Name for docker iamge, image:tag.
    # $2: Path volume
    sudo docker run -it --rm --privileged --runtime nvidia -p 10001:11001 -v $2 $1 /bin/bash
}
```

### Docker and NVIDIA-Container-Toolkit
A. Refer to <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker" title="NVIDIA Container-toolkit"> NVIDIA Container-toolkit</a> and <a href="https://github.com/vujadeyoon/Docker-Nvidia-Container-Toolkit" title="Docker-Nvidia-Container-Toolkit"> Docker-Nvidia-Container-Toolkit</a>.

B. Install Docker
```bash
usrname@hostname:~/path_curr$ curl https://get.docker.com | sh && sudo systemctl --now enable docker
```

C. Install NVIDIA Container Toolkit
```bash
# Setup the stable repository and the GPG key.
usrname@hostname:~/path_curr$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
usrname@hostname:~/path_curr$ sudo apt-get update

# Install nvidia-docker2 package.
usrname@hostname:~/path_curr$ sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon to complete the installation after setting the default runtime.
usrname@hostname:~/path_curr$ sudo systemctl restart docker
```

D. Test to run a base CUDA container
```bash
usrname@hostname:~/path_curr$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
```bash
    Wed Aug 25 04:34:07 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
    | 23%   35C    P8    11W / 250W |    364MiB / 12192MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+
```

### AWS CLI version 2
A. Refer to <a href="https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html" title="AWS CLI version 2"> AWS CLI version 2</a>.

B. Install AWS CLI version 2.
```bash
usrname@hostname:~/path_curr$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
usrname@hostname:~/path_curr$ unzip awscliv2.zip
usrname@hostname:~/path_curr$ sudo ./aws/install
usrname@hostname:~/path_curr$ aws configure
```
```bash
    AWS Access Key ID [None]: <access_key_id>
    AWS Secret Access Key [None]: <secret_access_key>
    Default region name [None]: ap-northeast-2
    Default output format [None]: json
```


## 15. Citrix <a name="citrix"></a>
A. How to install the Citrix.
* Download and install the Citrix Workspace App (e.g. <a href="https://www.citrix.com/downloads/workspace-app/linux/workspace-app-for-linux-latest.html" title="Citrix Workspace app 2108 for Linux"> Citrix Workspace app 2108 for Linux</a>) for Debian Packages - Full Packages (Self-Service Support).
```bash
    Do want to install the app protection component? (Yes)
```
* Then copy some certification files to the Citrix directory.
```bash
usrname@hostname:~/path_curr$ sudo cp -r /etc/ssl/certs/* /opt/Citrix/ICAClient/keystore/cacerts/
```

B. How to return to the local OS (i.e. Ubuntu MATE 20.04 LTS).
```bash
Ctrl + F2
```

C. How to fix a bug related to the user of citrixlog.
```bash
usrname@hostname:~/path_curr$ vi /etc/passwd
# Remove out the below codes.
citrixlog:x:1001:1001::/var/log/citrix:/bin/sh
:wq
```


## 16. File system <a name="file_system"></a>
### NTFS
A. How to fix NTFS disk write-protect.
```bash
usrname@hostname:~/path_curr$ sudo ntfsfix /dev/sdb1
```

### exFAT
A. How to enable a file system, exFAT.
```bash
usrname@hostname:~/path_curr$ sudo apt-get install exfat-utils exfat-fuse
```

## 17. File mode <a name="file_mode"></a>
A. How to change the mode of the given current directory recursively.
* Recommend mode of file: 755.
```bash
usrname@hostname:~/path_curr$ find ./ -type d -exec chmod -R -v 755 {} \;
```

B. How to change the mode of the given current file recursively.
* Recommend mode of file: 644.
```bash
usrname@hostname:~/path_curr$ find ./ -type f -exec chmod -R -v 644 {} \;
```
