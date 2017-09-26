require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';

model = require 'basic_model.lua'
print(model)

--load MNIST data
trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28)
trainlabels = mnist.traindataset().label+1
N = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):reshape(10000,1,28,28)
testlabels = mnist.testdataset().label+1
teSize = mnist.testdataset().size
print(N,teSize)

--print(trainData[1])
--image.save('test.jpg',trainData[1])
out = model:forward(trainData[1])

local theta,gradTheta = model:getParameters()
criterion = nn.ClassNLLCriterion()

local x,y

local feval = function(params)
	if theta~=params then
		theta:copy(params)
	end
	gradTheta:zero()
	local out = model:forward(x)
	--print(#x,#out,#y)
	local loss = criterion:forward(out,y)
	local gradLoss = criterion:backward(out,y)
	model:backward(x,gradLoss)
	return loss, gradTheta
end

--[[batchSize = 300
print('Training Starting')
local optimParams = {learningRate = 0.01, learningRateDecay = 0.0001}
local _,loss 
local losses = {}
for epoch=1,5 do
	collectgarbage()
	print('Epoch '..epoch..'/5')
	for n=1,N, batchSize do
		x = trainData:narrow(1,n,batchSize)
		y = trainlabels:narrow(1,n,batchSize)
		--print(y)
		_,loss = optim.sgd(feval,theta,optimParams)
		losses[#losses + 1] = loss[1]
	end
	local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
	gnuplot.pngfigure('Training2.png')
	gnuplot.plot(table.unpack(plots))
	gnuplot.ylabel('Loss')
	gnuplot.xlabel('Batch #')
	gnuplot.plotflush()

	--permute training data
	trainData = trainData:index(1,torch.randperm(trainData:size(1)):long())
end
--]]


local trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = 0.02
trainer.learningRateDecay = 0.001
trainer.shuffleIndices = 0
trainer.maxIteration = 20
batchSize = 300;

collectgarbage()
local iteration =1;
local currentLearningRate = trainer.learningRate;
local input=torch.Tensor(batchSize,1,28,28);
local target=torch.Tensor(batchSize);
local errorTensor = {}
trSize = N
print(trSize, trSize/batchSize);
print("Training starting")
while true do
	local currentError_ = 0
    for t = 1,math.floor(trSize/batchSize) do
    	local currentError = 0;
      	for t1 = 1,batchSize do
      		t2 = (t-1)*batchSize+t1;
        	target[t1] = trainlabels[t2];
        	input[t1] = trainData[t2];
			--print(t1)
        end
        currentError = currentError + criterion:forward(model:forward(input), target)
        --print(currentError)
		currentError_ = currentError_ + currentError*batchSize;
 		model:updateGradInput(input, criterion:updateGradInput(model:forward(input), target))
 		model:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
 		print("batch "..t.." done ==>");
 		collectgarbage()
    end
    ---- training on the remaining images, i.e. left after using fixed batch size.
    if(trSize%batchSize ~=0) then
	    local residualInput = torch.Tensor(trSize%batchSize,1,28,28);
	    local residualTarget = torch.Tensor(trSize%batchSize);

	    for t1=1,(trSize%batchSize) do
	    	t2=batchSize*math.floor(trSize/batchSize) + t1;
	    	residualTarget[t1] = trainlabels[t2];
	    	residualInput[t1] = trainData[t2];
		end
		currentError_ = currentError_ + criterion:forward(model:forward(residualInput), residualTarget)*(trSize%batchSize)
		--print("_ "..currentError_);
 		model:updateGradInput(residualInput, criterion:updateGradInput(model:forward(residualInput), residualTarget))
 		model:accUpdateGradParameters(residualInput, criterion.gradInput, currentLearningRate)
 		collectgarbage()
	end
	currentError_ = currentError_ / trSize
	print("#iteration "..iteration..": current error = "..currentError_);
	errorTensor[iteration] = currentError_;
	iteration = iteration + 1
  	currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
  	if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
    	print("# StochasticGradient: you have reached the maximum number of iterations")
     	print("# training error = " .. currentError_)
     	break
  	end
  	collectgarbage()
end


print('Testing accuracy')
correct = 0
class_perform = {0,0,0,0,0,0,0,0,0,0}
class_size = {0,0,0,0,0,0,0,0,0,0}
classes = {'0', '1', '2','3', '4','5', '6','7', '8','9'}
for i=1,teSize do
    local groundtruth = testlabels[i]
    local example = torch.Tensor(1,28,28);
    example = testData[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(example)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(#example,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end

torch.save('model_MNIST2.t7',model)
