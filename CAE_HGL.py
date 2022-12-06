import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from CAE import *
from HGL import *
from utils import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(8)  # 8
data_dir = 'mnist_data'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = train_transform
m = len(train_dataset)

# plt.imshow(train_data[0][0][0,:,:])
# plt.show()
batch_size = 512

train_loader = torch.utils.data.DataLoader(train_dataset+test_dataset, batch_size=batch_size, shuffle=True)


# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define an optimizer (both for the encoder and the decoder!)
lr = 0.01

# Initialize the two networks
d = 5

# model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d)
hgl = HGL(dim=d, units=30, init_mode='randn')  # 5,30效果最好
decoder = Decoder(encoded_space_dim=d)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': hgl.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
hgl.to(device)
decoder.to(device)


### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        # image_noisy = add_noise(image_batch, noise_factor)
        image_noisy = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_noisy)
        # print(encoded_data)
        # ss
        # Decode data
        hgl_data, cl = hgl(encoded_data)  # cl:聚类label

        decoded_data = decoder(encoded_data)
        # Evaluate loss
        hgl_loss = F.mse_loss(hgl_data.squeeze(1), 1.2*torch.ones(encoded_data.shape[0]).to(device))
        loss = loss_fn(decoded_data, image_noisy)
        # Backward pass
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for param in encoder.parameters():
            param.requires_grad = False
        hgl_loss.backward()
        for param in encoder.parameters():
            param.requires_grad = True
        optimizer.step()
        # Print batch loss
        train_loss.append(loss.detach().cpu().numpy())
    print('\t partial train loss (single batch): %f' % (hgl_loss.data))

    return np.mean(train_loss)


# Val function
def val_epoch(encoder, decoder, device, dataloader):
    # Set train mode for both the encoder and the decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        true_label = []
        cluster_label = []
        for image_batch, label in dataloader:
            image_noisy = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            hgl_data, cl = hgl(encoded_data)  # cl:聚类label

            true_label.append(label.cpu())
            cluster_label.append(cl.cpu())
        true_label = torch.cat(true_label)
        cluster_label = torch.cat(cluster_label)

        # confusion = confusion_matrix(true_label, cluster_label)
        confusion = torch.zeros(10, 30)
        for x, y in zip(true_label, cluster_label):
            confusion[x, y] += 1
        print('===' * 10)
        print(torch.max(confusion, dim=0)[0] / torch.sum(confusion, dim=0))
        print(torch.mean(torch.max(confusion, dim=0)[0] / torch.sum(confusion, dim=0)))
        print(torch.sum(torch.max(confusion, dim=0)[0] / torch.sum(torch.sum(confusion, dim=0)), dim=0))
        print(confusion.shape)
        print('===' * 10)
        display(confusion)
        Confusion_matrix_show(Clustering_integration(confusion))
        # print(true_label[10:33])
        # print(cluster_label[10:33])
    return np.mean(train_loss)


def plot_ae_outputs(encoder, decoder, n=10):
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = train_dataset[i][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.draw()
    plt.pause(0.05)


num_epochs = 75
diz_loss = {'train_loss': [], 'val_loss': []}
for epoch in range(num_epochs):
    train_loss = train_epoch(encoder, decoder, device,
                             train_loader, loss_fn, optim)
    print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs, train_loss))
    diz_loss['train_loss'].append(train_loss)
    if epoch % 15 == 14:
        val_epoch(encoder, decoder, device, train_loader)
    if epoch == 74:
        TSNE_SHOW(encoder, train_loader)
    plot_ae_outputs(encoder, decoder, n=10)
plt.ioff()
plt.show()
