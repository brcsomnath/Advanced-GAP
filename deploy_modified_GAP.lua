require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';

model = torch.load('model_MNIST2.t7')

model:remove(13)
model:remove(12)
model:remove(11)
model:remove(10)
model:remove(9)
model:remove(8)
model:remove(7)
model:remove(6)
model:remove(3)

err_tensor = torch.load('modifiedGAP_errTensor.t7')
inp = torch.Tensor(1,28,28)
out = model:forward(inp)
numFilters = out:size(1)
classes = {"0","1","2","3","4","5","6","7","8","9"}
--[[Weights = torch.Tensor(numFilters,#classes)
for f=1,numFilters do
	Weights[f] = err_tensor[f]-err_tensor[numFilters+1]
end
--]]
Weights = torch.Tensor(numFilters+1)
for f=1,numFilters+1 do
	Weights[f] = err_tensor[f]:sum()
end

print('loading image')
img = image.load('numbers9.jpg',1)
img = img:double():reshape(1,img:size(1),img:size(2))

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
local CAM = (Weights[1]-Weights[1+numFilters])*filters[1]
for f=2,numFilters do
	CAM = CAM + (Weights[f]-Weights[1+numFilters])*filters[f]
	collectgarbage()
end
CAM:add(-CAM:min())
CAM:div(CAM:max())
--CAM = image.scale(CAM,"*2")
image.save('modified_GAP_results/numbers9_filAll.jpg',CAM)
collectgarbage()
