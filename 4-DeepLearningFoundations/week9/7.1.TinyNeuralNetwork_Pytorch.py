import torch
from torch import nn

# ======================================================================
# 1) Tiny dataset
# ======================================================================
# We use 2 input features and 1 binary target.
# Each row is one example.
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.2, 0.1],
    [0.1, 0.9],
    [0.8, 0.2],
    [0.9, 0.8],
], dtype=torch.float32)

# Binary targets must be float for BCEWithLogitsLoss.
# Shape is [n_samples, 1]
y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
], dtype=torch.float32)

# ======================================================================
# 2) Tiny Model
# ======================================================================
class Tinynnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2,4), # input : 2 features, 4 neurons. 
            nn.ReLU(),
            nn.Linear(4,1) #Output : 1 logit
        )

    def forward(self, x):
        return self.net(x)
        # Following is what happens underneath this simple function 
        # (Calcualte the weighted sum+bias, apply non-linear function and capture the output)
        # z1 = self.hidden(x)
        # a1 = self.relu(z1)
        # z2 = self.output(a1)
        # return z2

model = Tinynnet()

# ======================================================================
# 3) Loss and Optimizer
# ======================================================================
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ======================================================================
# 4) Training loop
# ======================================================================
num_epochs = 600

for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad() # To clear out gradient accumulations
    logits = model(X) #Forward pass
    loss = loss_fn(logits,y) #Calculate loss
    loss.backward() # Run backpropagation to calcuate gradients
    optimizer.step() # Update parameters

    if(epoch==0):
        print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    if (epoch + 1) % 20 == 0:
        print(f"epoch={epoch+1:03d} loss={loss.item():.4f}")

    # print(f"epoch={epoch:03d} loss={loss.item():.4f}")
    
''' -- OUTPUT --
epoch=000 loss=0.6101
epoch=020 loss=0.5470
epoch=040 loss=0.5148
epoch=060 loss=0.4814
epoch=080 loss=0.4412
epoch=100 loss=0.3951
epoch=120 loss=0.3486
epoch=140 loss=0.3042
epoch=160 loss=0.2635
epoch=180 loss=0.2277
epoch=200 loss=0.1970
epoch=220 loss=0.1710
epoch=240 loss=0.1491
epoch=260 loss=0.1308
epoch=280 loss=0.1154
epoch=300 loss=0.1025
epoch=320 loss=0.0915
epoch=340 loss=0.0822
epoch=360 loss=0.0742
epoch=380 loss=0.0673
epoch=400 loss=0.0614
epoch=420 loss=0.0562
epoch=440 loss=0.0517
epoch=460 loss=0.0477
epoch=480 loss=0.0442
epoch=500 loss=0.0411
epoch=520 loss=0.0383
epoch=540 loss=0.0358
epoch=560 loss=0.0336
epoch=580 loss=0.0316
epoch=600 loss=0.0298
'''
# ======================================================================
# 5) Prediction Phase
# ======================================================================
model.eval()

with torch.no_grad():
    logits=model(X)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()


print("\nProbabilities:")
print(probs)

print("\nPredictions:")
print(preds)

print("\nTrue labels:")
print(y)

''' --OUTPUT--
Probabilities:
tensor([[6.7597e-05],
        [2.6432e-02],
        [3.7721e-02],
        [9.9068e-01],
        [2.2318e-04],
        [2.7394e-02],
        [3.5143e-02],
        [9.0560e-01]])

Predictions:
tensor([[0.],
        [0.],
        [0.],
        [1.],
        [0.],
        [0.],
        [0.],
        [1.]])

True labels:
tensor([[0.],
        [0.],
        [0.],
        [1.],
        [0.],
        [0.],
        [0.],
        [1.]])

'''