"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde


fname_train = "../dataset/dataset.train"
fname_test = "../dataset/dataset.test"

# 创建 DataSet 对象 for handling the dataset. This includes loading the dataset, preparing it for the model, and standardizing the features. 
data = dde.data.DataSet(
    fname_train=fname_train,  # Path to the training data file
    fname_test=fname_test,    # Path to the testing data file
    col_x=(0,),               # Column index that contains the inputs (features)
    col_y=(1,),               # Column index that contains the outputs (labels)
    standardize=True          # Standardize features (zero mean and unit variance)
)

# 定义 神经网络
# - Layer sizes:         input layer of 1 node, three hidden layers each of 50 nodes, and an output layer of 1 node.
# - Activation function: 'tanh' (hyperbolic tangent) used for non-linear transformation.
# - Weight initializer: 'Glorot normal' (also known as Xavier normal), which is good for keeping the scale of gradients roughly the same in all layers.
layer_size = [1] + [50] * 3 + [1]
print('layer_size = ', layer_size)
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer) # 创建 the feed-forward neural network.

# 创建 a model by combining the data and the neural network. This model will be used for training.
model = dde.Model(data, net)
# Compile the model with the Adam optimizer and a learning rate of 0.001. Also, specify 'l2 relative error' as a metric for evaluation.
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
# Train the model for a total of 50,000 iterations and capture the loss history and training state.
losshistory, train_state = model.train(iterations=50000)

# Save and plot the training progress. This function saves the plots of loss history and state of the model training.
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
