import matplotlib.pyplot as plt
from torchvision.utils import make_grid #to make grids of images 

def show(tensor,ch=1,size =(28,28),num=25):
  # tensor -- 128 * 784 ( 128 = Batch_size, 28*28) 
  data = tensor.detach().cpu().view(-1,ch,*size) # No need to calculate the gradients it is in visualziaton mode
  # data = 128 *1 * 28 * 28
  grid = make_grid(data[:num],nrow = 5).permute(1,2,0) #out of 128 we can take 25 images for visualziatiom
  # permute use to rotate the axis [data will give 25 * 1 * 28 * 28]
  # permute basically need to change the order of channels In matplotlib they use H*W*C but pyTorch use C*H*W
  plt.imshow(grid) # to show the grid 
  plt.show() # to show the grid