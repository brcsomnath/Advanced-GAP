require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';

model = torch.load('model_MNIST2.t7')

model2 = nn.Sequential()
model2:insert(model:get(14),1)
model:remove(14)
model2:insert(model:get(13),1)
model:remove(13)
model2:insert(model:get(12),1)
model:remove(12)
model2:insert(model:get(11),1)
model:remove(11)
model2:insert(model:get(10),1)
model:remove(10)

print(model)
print(model2)

checkOut = model:forward(torch.rand(1,28,28))
numFilters = checkOut:size(1)

trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28)
trainlabels = mnist.traindataset().label+1
trSize = mnist.traindataset().size

sorted,indices = torch.sort(trainlabels)
classes = {'0', '1', '2','3', '4','5', '6','7', '8','9'}
classSize = torch.Tensor(#classes):zero()
for i=1,trSize do
	classSize[trainlabels[i]] = classSize[trainlabels[i]]+1
end
print(classSize)
print(classSize:sum())

criterion = nn.ClassNLLCriterion()

errorTensor = torch.Tensor(1+numFilters,#classes):zero()

batchSize = 200
indexDone=0
errorOccurred = 0

for c=1,#classes do
	print('Starting Class '..c)
	for t=1,classSize[c]/batchSize do
		local input = torch.Tensor(batchSize,1,28,28)
		local target = torch.Tensor(batchSize)
		classTarget = c
		for t1 = 1,batchSize do
			local t2 = (t-1)*batchSize + t1 + indexDone
			input[t1] = trainData[indices[t2]]
			target[t1] = trainlabels[indices[t2]]
			if target[t1]~=c then
				print('error at c='..c..' t='..t..' t1='..t1)
				errorOccurred=1
				break;
			end
		end
		if errorOccurred==1 then 
			break 
		end
		local filters = model:forward(input)
		local error_ = criterion:forward(model2:forward(filters),target)*batchSize
		errorTensor[numFilters+1][classTarget] = errorTensor[numFilters+1][classTarget] + error_
		for f=1,numFilters do
			collectgarbage()
			local filter_masked = torch.Tensor(filters:size()):copy(filters)
			filter_masked[{{},{f},{},{}}]:zero()
			local error_masked = criterion:forward(model2:forward(filter_masked),target)*batchSize
			errorTensor[f][classTarget] = errorTensor[f][classTarget] + error_masked
		end
	end
	if errorOccurred==1 then 
		break 
	end
	if classSize[c]%batchSize~=0 then
		local remaining = classSize[c]%batchSize
		local input = torch.Tensor(remaining,1,28,28)
		local target = torch.Tensor(remaining)
		classTarget = c
		for t1 = 1,remaining do
			local t2 = classSize[c]-remaining + t1 + indexDone
			input[t1] = trainData[indices[t2]]
			target[t1] = trainlabels[indices[t2]]
			if target[t1]~=c then
				print('error at c='..c..' t='..t..' t1='..t1)
				errorOccurred=1
				break;
			end
		end
		if errorOccurred==1 then 
			break 
		end
		local filters = model:forward(input)
		local error_ = criterion:forward(model2:forward(filters),target)*remaining
		errorTensor[numFilters+1][classTarget] = errorTensor[numFilters+1][classTarget] + error_
		for f=1,numFilters do
			collectgarbage()
			local filter_masked = torch.Tensor(filters:size()):copy(filters)
			filter_masked[{{},{f},{},{}}]:zero()
			local error_masked = criterion:forward(model2:forward(filter_masked),target)*remaining
			errorTensor[f][classTarget] = errorTensor[f][classTarget] + error_masked
		end
	end
	if errorOccurred==1 then 
		break 
	end
	indexDone = indexDone+classSize[c]
	--print(errorTensor)
end


torch.save('modifiedGAP_errTensor.t7',errorTensor)