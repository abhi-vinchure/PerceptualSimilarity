import argparse
import os
import lpips
from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-d2','--dir2', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('-net', '--net', type = str, default = 'squeeze')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net=opt.net,version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)

dist01_sum = 0
dist02_sum = 0
for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = lpips.load_image(os.path.join(opt.dir0,file))
 
		img0 = lpips.im2tensor(img0) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
		img2 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir2,file)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()
			img2 = img2.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		dist01_sum += dist01.item()
		#map file to dcgan image distance to real
		dist01_dict = {"dcgan " + file: dist01}
		dist02 = loss_fn.forward(img0,img2)
		dist02_sum += dist02.item()

		#map file to ganmlp image distance to real
		dist02_dict = {"mlpgan " + file: dist02}
		print(dist01_dict)
		print(dist02_dict)
		#print('%s: %.3f'%(file,dist01))
		
		#print('%s: %.3f'%(file,dist02))
	
		#f.writelines('%s: %.6f\n'%(file,dist01))
print("DCGAN = ", dist01_sum/6000)
print("GANMLP = ", dist02_sum/6000)

f.close()
