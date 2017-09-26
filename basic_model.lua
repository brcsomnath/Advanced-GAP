require 'nn'

model = nn.Sequential()
model:add(nn.SpatialConvolution(1,6,5,5,1,1,2,2)) --1
model:add(nn.ReLU()) --2
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --3
-- size: 

model:add(nn.SpatialConvolution(6,16,5,5,1,1,2,2)) --4
model:add(nn.ReLU()) --5
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --6
-- size: 

model:add(nn.View(-1):setNumInputDims(3)) --7
model:add(nn.Linear(784,120)) --8
model:add(nn.ReLU()) --9
model:add(nn.Linear(120,84)) --10
model:add(nn.ReLU()) --11
model:add(nn.Linear(84,10)) --12
model:add(nn.LogSoftMax()) --13
--]]
return model