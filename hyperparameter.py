#hyperparame

# for Model
image_size = 64 # input Image Size 
s16 = image_size // 16
gf_dim = 64	# Generator conv layer 
df_dim = 64    # Dimension conv layer 
c_dim = 3    	# n_color 3
z_dim = 100 	# input Noise dim 
lr = 0.0002
beta1 = 0.5
output_size = 64  # output Image Size 

# for Train
epoch = 20
batch_size = 64
sample_size = 64
# data & save Path
datasets_path = "../../dataset/celebA_crop/*.jpg"
samples_path = "./samples"
checkpoint_path = "./ckpt"
log_path = "./board/log"

# Data_Processing
crop_size = 108
is_train = True
is_crop = True

#  Don't maind
num_files = None
batch_steps = None