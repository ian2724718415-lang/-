import numpy as np
import matplotlib.pyplot as plt
SEED = 42
np.random.seed(SEED)
LEARNING_RATE = 0.01
EPOCHS = 200
BATCH_SIZE = 32
N_SAMPLES = 1000
N_FEATURES = 5
def generate_data(n_samples=N_SAMPLES,n_features=N_FEATURES,noise=0.1):
    true_w=np.array([1.5,-2.0,0.8,-1.2,0.5]).reshape(-1,1)
    X =np.hstack([np.ones((n_samples,1)),np.random.randn(n_samples,n_features-1)])
    y = X @ true_w + np.random.randn(n_samples,1)
    split_idx = int(0.8 * n_samples)
    X_train,X_val = X[:split_idx],X[split_idx:]
    y_train,y_val = y[:split_idx],y[split_idx:]
    return X_train,y_train,X_val,y_val,true_w
def mse_loss(y_pred,y_true):
    n=len(y_true)
    return np.sum((y_pred - y_true)**2)/n
def full_batch_gd(X_train,y_train,X_val,y_val,lr=LEARNING_RATE,epochs=EPOCHS):
    n_samples,n_features = X_train.shape
    w = np.zeros((n_features,1))
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        y_pred_train = X_train @ w
        grad = (2 / n_samples) * (X_train.T @ (y_pred_train - y_train))
        w = w-lr*grad
        train_loss = mse_loss(y_pred_train,y_train)
        val_loss =mse_loss(X_val @ w,y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if(epoch + 1)% 20 == 0:
            print(f"Full-batch | Epoch {epoch+1}: Train Loss = {train_loss:.6f},Val Loss = {val_loss:.6f}")
    return w, train_losses,val_losses
def mini_batch_gd(X_train,y_train,X_val,y_val,lr=LEARNING_RATE,epochs=EPOCHS,batch_size=BATCH_SIZE):
    n_samples,n_features = X_train.shape
    w = np.zeros((n_features,1))
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        for i in range(0,n_samples,batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            batch_size_actual = len(batch_X)
            y_pred_batch = batch_X @ w
            grad_batch = (2 / batch_size_actual)*(batch_X.T @ (y_pred_batch - batch_y))
            w = w - lr * grad_batch
        y_pred_train_full = X_train @ w
        train_loss = mse_loss(y_pred_train_full,y_train)
        val_loss = mse_loss(X_val @ w,y_val)
        train_losses.append(val_loss)
        if (epoch + 1) % 20 ==0:
            print(f"Mini-batch | Epoch{epoch+1}: Train Loss = {train_loss:.6f},Val Loss = {val_loss:.6f}")
    return w,train_losses,val_losses
if __name__=="__main__":
    X_train,y_train,X_val,y_val,true_w= generate_data()
    print("===== 全量梯度下降训练 =====")
    w_full,train_losses_full,val_losses_full = full_batch_gd(X_train,y_train,X_val,y_val)
    np.random.seed(SEED)
    print("\n===== 小批量梯度下降训练 =====")
    w_mini,train_losses_mini,val_losses_mini=mini_batch_gd(X_train,y_train,X_val,y_val)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses_full,label="Full-batch GD",linewidth=2)
plt.plot(train_losses_mini,label="Mini-batch GD",linewidth=2,alpha=0.8)
plt.xlabel("Epochs")
plt.ylabel("Train MSE Loss")
plt.title("Train Loss Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.subplot(1,2,2)
plt.plot(val_losses_full,label="Full-batch GD",linewidth=2)
plt.plot(val_losses_mini,label="Mini-batch GD",linewidth=2,alpha=0.8)
plt.xlabel('Epochs')
plt.ylabel('Val MSE Loss')
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("\===== 最终参数比 =====")
print(f"真实参数：{true_w.flatten()}")
print(f"Full-batch GD 参数:{w_full.flatten()}")
print(f"Mini-batch GD 参数:{w_mini.flatten()}")