from train import *

data_dir = './crops'
save_dir = './save'
patchsize = 256
batchsize = 32
n_channel = 3
n_feature = 48
n_epoch = 100
lr = 3e-4
gamma = 0.5
Lambda1 = 1.0
Lambda2 = 1.0
ckpt_freq = 10

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Training Set
TrainingDataset = DataLoader_Imagenet_val(data_dir, patch=patchsize)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

# Network
network = UNet(in_nc=n_channel,
               out_nc=n_channel,
               n_feature=n_feature)
if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()

# about training scheme
num_epoch = n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=lr)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=gamma)
print("Batchsize={}, number of epoch={}".format(batchsize, n_epoch))

checkpoint(network, 0, "model")
print('init finish')

for epoch in range(1, n_epoch + 1):
    cnt = 0

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()
    for iteration, noisy in enumerate(TrainingLoader):
        st = time.time()
        noisy = noisy / 255.0
        noisy = noisy.cuda()
#         noisy = noise_adder.add_train_noise(clean)

        optimizer.zero_grad()

        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        with torch.no_grad():
            noisy_denoised = network(noisy)
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = network(noisy_sub1)
        noisy_target = noisy_sub2
        Lambda = epoch / opt.n_epoch * opt.increase_ratio
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        loss1 = torch.mean(diff**2)
        loss2 = Lambda * torch.mean((diff - exp_diff)**2)
        loss_all = Lambda1 * loss1 + Lambda2 * loss2

        loss_all.backward()
        optimizer.step()
        print(
            '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
            .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                    np.mean(loss2.item()), np.mean(loss_all.item()),
                    time.time() - st))
    checkpoint(network, epoch, "model")

    scheduler.step()