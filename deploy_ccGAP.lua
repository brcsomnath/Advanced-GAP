require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';

model = torch.load('model_MNIST2.t7')

model:remove(12)
model:remove(11)
model:remove(10)

model:remove(9)
model:remove(6)
model:remove(3)

Weights = torch.load('ccGAP_errTensor.t7')
inp = torch.Tensor(1,28,28)
out = model:forward(inp)
numFilters = out:size(1)
classes = {"0","1","2","3","4","5","6","7","8","9"}
--[[Weights = torch.Tensor(numFilters,#classes)
for f=1,numFilters do
	Weights[f] = err_tensor[f]-err_tensor[numFilters+1]
end
--]]

print('loading image')
img = image.load('ccGAP Results/numbers9.jpg',1)
img = img:double():reshape(1,img:size(1),img:size(2))
img = 1-img
filters = model:forward(img)

--[[for i=1,#classes do
	local CAM = torch.Tensor(filters:size(2),filters:size(3))
	local CAM = Weights[1][i]*filters[1]
	for j=2,filters:size(1) do
		CAM = CAM + Weights[j][i]*(filters[j])
	end
	CAM:add(-CAM:min())
	CAM:div(CAM:max())
	CAM = image.scale(CAM,"*2")
	image.save('modified_GAP_results/fb_fil'..i..'.jpg',CAM)
	collectgarbage()
end
--]]

local CAM = torch.Tensor(filters:size(2),filters:size(3))
local CAM = (Weights[1]:sum())*filters[1]
for f=2,numFilters do
	CAM = CAM + (Weights[f]:sum())*filters[f]
	collectgarbage()
end
CAM:add(-CAM:min())
CAM:div(CAM:max())
mean_val = CAM:mean();
thresh_level = 2;
std_val = CAM:std();
CAM_thresh = torch.clamp(CAM,mean_val+thresh_level*std_val,1)
CAM_thresh = CAM_thresh- torch.cmul(CAM_thresh:eq(mean_val+thresh_level*std_val):double(),CAM_thresh);
print(CAM:min(),CAM:max(),CAM:mean(),CAM:std())
--CAM = image.scale(CAM,"*2")
image.save('ccGAP Results/numbers9_filAll.jpg',CAM_thresh)
collectgarbage()
