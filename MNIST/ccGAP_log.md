# Training a network on MNIST ->
## Testing accuracy
Overall correct 9878 percentage correct 98.78 % <br/>
0	98.877551020408 % <br/>
1	99.383259911894 % <br/>
2	99.22480620155 % 	<br/>
3	99.009900990099 % 	<br/>
4	99.49083503055 % 	<br/>
5	98.654708520179 % 	<br/>
6	98.12108559499 % 	<br/>
7	98.929961089494 % 	<br/>
8	99.17864476386 % 	<br/>
9	96.828543111992 % <br/>

## Network architecture ->
nn.Sequential { <br/>
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output] <br/>
  (1): nn.SpatialConvolution(1 -> 6, 5x5, 1,1, 2,2) <br/>
  (2): nn.ReLU<br/>
  (3): nn.SpatialMaxPooling(2x2, 2,2)<br/>
  (4): nn.SpatialConvolution(6 -> 16, 5x5, 1,1, 2,2)<br/>
  (5): nn.ReLU<br/>
  (6): nn.SpatialMaxPooling(2x2, 2,2)<br/>
  (7): nn.SpatialConvolution(16 -> 32, 3x3, 1,1, 1,1)<br/>
  (8): nn.ReLU<br/>
  (9): nn.SpatialMaxPooling(2x2, 2,2)<br/>
  (10): nn.View(288)<br/>
  (11): nn.Linear(288 -> 10)<br/>
  (12): nn.LogSoftMax<br/>
}
---
