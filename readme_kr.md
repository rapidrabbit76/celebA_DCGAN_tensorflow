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

1. dl.py 를 실행시켜 celebA datasets를 다운로드 합니다
dl.py code from [here](https://github.com/tensorlayer/dcgan/blob/master/download.py)

        $ python3 dl.py celebA 

2. 학습전 데이터를 먼저 처리 하기위해서 "dataset/data_process.py" 를 열어 [IMAGE_PATH,SAVE_PATH,Thread_core] 를 설정해 데이터셋의 경로를 지정하고 cpu 코어수를 입력한다음 **data_process.py** 를 실행시킵니다. 파일수가 있기 때문에 시간이 걸립니다... 

        $ python3 data_process.py

3. hyperparameter.py 열어서 경로를 환경에 맞게 설정합니다.
    
        datasets_path = "전처리된 데이터 경로를 입력하시면 됩니다."
        samples_path = "./samples"
        checkpoint_path = "./ckpt"
        log_path = "./board/log"
    
4. hyperparameter.py 에서 하이퍼파라미터를 설정하시고 train.py 를 실행시켜 학습을 진행시켜주시면 됩니다.

        $ python3 train.py

5. ckpt(체크포인트) 값을 불러오기위해선 train.py 를 열어 원하시는 체크포인트 경로를 입력시켜주시면 됩니다. 예를 들어 model.load(**"./ckpt/DCGAN.ckpt-76500"**) 식으로 입력 해주시면 됩니다. 

        load, global_steps = model.load("Put the path here")
        

6. 학습과정을 텐서보드로 확인하시고 싶으면 **"./board"** 를 경로값으로 입력하시면 됩니다. 

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