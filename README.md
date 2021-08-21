### Technologies
- Python
- Docker
- Tensorflow
- CUDA
- CONDA
- Tkinter



### APP Features

- Social Distancing Detection using area of interest in the image and warp perspective
- Using Keras Retinanet as people detector 
- People counter in real time 




# Sample 

	![](https://i.ibb.co/2YNMgcd/campanario-out.gif)

# Instructions 
- Download the weights from this link https://drive.google.com/drive/folders/1DUk-Hg3ZYAFzi8fyKZpfITzNhsdzfPWp?usp=sharing and put it in the main folder
- There are 2 ways to use this project: 
    - Install all requirements locally.
    - Use dockerfile or the image in docker hub.
- Conda installation
    - We are using conda because the easy way to install tensorflow-gpu (CUDA), but if you want to install your own environment in other like pyenv, venv, etc, its ok. 
    - Inside the main folder you can see two files that their names are "ambiente_windows.txt" and "ambiente_linux_server.txt", if you are working in windows platflorm, you need to install "ambiente_windows.txt", but if you have linux platform, use "ambiente_linux_server.txt"
        - Open the command line (terminal or CMD in windows), and go to the project directory.
        - conda create -n <env_name> --file ambiente_windows.txt
        - conda activate <env_name>
        - pip install -r requirements.txt
    - After the installation, you need to clone the Keras RetinaNet Project. 
        - Make sure that you are in the project main directory.
        - git clone https://github.com/fizyr/keras-retinanet.git 
        - git checkout 42068ef9e406602d92a1afe2ee7d470f7e9860df
        - python setup.py build_ext --inplace
        - pip install . --user
        - Thats it. 
- Docker installation
    - If you just test the project, use Docker installation. 
    - Make sure that you are in the project main directory.
    - As you can see, there is Dockerfile in the folder. 
    - First of all, install Docker Desktop in your machine (avoid it, if you have been installed docker). 
        - docker build -t <image_name> .
        - Wait while image is building. 
        - If you are working with Windows 10, you need to install Xming (https://sourceforge.net/projects/xming/)
        - After the installation, open your CMD, and write ipconfig (just for windows).
        - save your IP Address (ipv4).
        - set-variable -name DISPLAY -value <ip_adress>:0.0
        - after that, you can run the docker. 
        - docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw  <image_name>
        - Find campanario.mp4 video in the main folder. 
        - Enjoy it. 
    - Use Dockerhub image 
        - docker pull daigen/social_distancing_ai
        - If you are working with Windows 10, you need to install Xming (https://sourceforge.net/projects/xming/)
        - After the installation, open your CMD, and write ipconfig (just for windows).
        - save your IP Address (ipv4).
        - set-variable -name DISPLAY -value <ip_adress>:0.0
        - after that, you can run the docker. 
        - docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw  <image_name>
        - Find campanario.mp4 video in the main folder. 
        - Enjoy it. 

# Use
    - python video_detection.py --video <video_test> --output <optional output filename>
    - Go to the GUI directory.
        - python gui.py
        
 ![](https://i.ibb.co/mtYrPqM/gui-image.png)

### Notes
- This project has been configured to works in the known place (we know the real area size). Please, if you want to use it, you need to change the area of interest points inside the code to have an accurate precision. Maybe in the future, we could add an interactive interface to add parameters to define the area of interest according to any situation. 
- You should have NVIDIA GPU with CUDA installed in your os system to run docker image with GPU. If you have windows, you need to follow this guide https://drive.google.com/drive/folders/1DUk-Hg3ZYAFzi8fyKZpfITzNhsdzfPWp?usp=sharing

    





# Authors 
- Luis Felipe Tobar Sotelo
	[Linkedin](https://www.linkedin.com/in/luis-felipe-tobar-sot/)
	[Github](https://github.com/felipetobars/)

- Steven Parra Giraldo 
	[Linkedin](https://www.linkedin.com/in/stevenparragiraldo/)
	[Github](https://github.com/StraigenDaigen/)

- Natali Velandia Fajardo
	[Linkedin](https://www.linkedin.com/in/natali-velandia-60346715a/)



# Acknowledgments
- Juan Carlos Perafán
	[Linkedin](https://www.linkedin.com/in/juanperafan/)


### End
