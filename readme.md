# celebA_DCGAN
## [DCGAN](https://arxiv.org/abs/1511.06434) by Tensorflow 

![samples](./as/samples.png)

Result Image (18 epochs)

# Prerequisites
* python3 == 3.6.6
* tensorflow == 1.11.0 +
* tensorlayer == 1.10.1 +
* Scipy

# Usege
[[한국어]](./readme_kr.md)

1. run dl.py in dataset then download celebA datasets
dl.py code from [here](https://github.com/tensorlayer/dcgan/blob/master/download.py)

        $ python3 dl.py celebA 

2. Set [IMAGE_PATH,SAVE_PATH,Thread_core] in "dataset/data_process.py" and run (Crop & Resize)

        $ python3 data_process.py

3. open hyperparameter.py set PATHs
    
        datasets_path = "preprocessed dataPATH (SAVE_PATH/)"
        samples_path = "./samples"
        checkpoint_path = "./ckpt"
        log_path = "./board/log"
    
4. set hyperparameter.py and run train.py

        $ python3 train.py

5. if you want to load ckpt

        load, global_steps = model.load("Put the path here")

6. visualization of training process

        $ tensorboard --logdir ./board

# Model Architecture
## Generator
![alt](https://hyeongminlee.github.io/img/GAN_003/fig3.PNG)

## Discriminator
![Dis](https://hyeongminlee.github.io/img/GAN_003/fig4.PNG)

### Model Architecture Images From [taeoh-kim's GitHub](https://github.com/taeoh-kim/Pytorch_DCGAN)



# Result
![anime](./as/training_anime.gif)
### Images for each 500 step


# Problem
On average, if more than 20 epochs are exceeded, "mode collapse" is clearly displayed.

Sometimes "mode collapse" occurs while learning, and if the D_loss falls below 1e-2 there is almost can not recovery... 

At gif, you can see that the middle(step) image is sometimes worse.

I'm trying to fix it. 

## Things I've tried
***

1. Lower lr to 4e-5 and try 100 and Generator_model training twice, but "mode collapse" occurs between 60 and 70 of the epoch...

2. just Generator_model training twice "mode collapse" occurs between 13~15....