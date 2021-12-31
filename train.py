##################################################################################################
# 执行如下命令进行训练
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py

##################################################################################################


import argparse
import os
import tempfile
import torch.distributed as dist
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
import notify

writer = SummaryWriter("./log")


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'gloo'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    print(args.dist_backend)
    print(args.dist_url)
    print(args.world_size)
    print(args.rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device id (i.e. 2 or 2, 3 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--syncBN', type=bool, default=True)  ##可以不使用，这个对训练速度有影响
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    args = opt
    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    # learning_rate *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    # device_ids = [2]

    #   local_rank = parser.local_rank
    # if local_rank != -1:
    #     dist_backend = 'nccl'
    #     dist.init_process_group(backend=dist_backend)  # 初始化进程组，同时初始化 distributed 包
    # device = local_rank if local_rank != -1 else (
    #     torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    # torch.cuda.set_device(local_rank)  # 配置每个进程的gpu

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = torchvision.datasets.CIFAR10(root="datasets", train=True, transform=data_transform["train"], download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="datasets", train=False, transform=data_transform["val"], download=True)

    batch_size = 16
    lr = 0.00001
    epochs = 50

    nw = 0 # 原文使用：min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数
    print('Using {} dataloader workers every process'.format(nw))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               num_workers=nw,
                                               pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                sampler=test_sampler,
                                                num_workers=nw,
                                                pin_memory=True,
                                                drop_last=True)


    # 网络创建成功
    resnet = resnet34()# .to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(resnet)
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    if rank == 0:
        torch.save(resnet.state_dict(), checkpoint_path)
    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    resnet.load_state_dict(torch.load("model.pth", map_location=device))

    # resnet.load_state_dict(torch.load("model.pth"))

    # # 修改模型全连接层，输出10种分类，
    in_channel = resnet.fc.in_features
    resnet.fc = nn.Linear(in_channel, 10)

    resnet.to(device)  # 封装之前要把模型移到对应的gpu
    resnet = torch.nn.parallel.DistributedDataParallel(resnet, device_ids=[args.gpu],
                                                       find_unused_parameters=True)

    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致


    # resnet = torch.nn.DataParallel(resnet, device_ids=device_ids)  # 指定要用到的设备
    # resnet = resnet.cuda(device=device_ids[0])  # 模型加载到设备0
    # device_ids = list(map(int, args.device_ids.split(',')))
    # dist.init_process_group(backend='nccl', init_method='env://')
    # device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
    # torch.cuda.set_device(device)
    # model = resnet().to(device)
    # model = DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]],
    #                                 output_device=device_ids[args.local_rank])

    # define loss function
    loss_function = nn.CrossEntropyLoss()# .to(device)
    # loss_function = loss_function.cuda()# device=device_ids[0]
    # construct an optimizer
    #params = [p for p in resnet.parameters() if p.requires_grad]

    optimizer = optim.Adam(resnet.parameters(), lr=lr)

    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_dataloader)

    total_train_step = 0
    total_test_step = 0
    acc = 0.0
    for epoch in range(epochs):

        train_sampler.set_epoch(epoch=epoch)
        # train
        print("开始训练")
        resnet.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader)
        for step, data in enumerate(train_bar):

            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            #images = nn.DistributedSampler(images)
            #targets = nn.DistributedSampler(targets)
            # images = images.cuda()#.to(device)#device=device_ids[0]
            # targets = targets.cuda()#to(device)#device=device_ids[0]
            optimizer.zero_grad()
            output = resnet(images)


          #  print(type(output))
           # print(output.shape)
           # print(targets.shape)

            loss = loss_function(output, targets)
            loss.backward()
            loss = reduce_value(loss, average=True)
            optimizer.step()

            # print statistics
            running_loss += loss.item()


            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
            predict_y = torch.max(output, dim=1)[1]
            acc = torch.eq(predict_y, targets).sum().item() / batch_size
            if total_train_step % 10 == 0:
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                writer.add_scalar("train_accuracy", acc, total_train_step)

            total_train_step += 1


        # validate
        resnet.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_dataloader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                #val_images = nn.DistributedSampler(val_images)
                #val_labels = nn.DistributedSampler(val_labels)
                # val_images = val_images.cuda() # to(device)device=device_ids[0]
                # val_labels = val_labels.cuda()# to(device)device=device_ids[0]
                outputs = resnet(val_images)
                val_labels = val_labels.to(device)
                loss = loss_function(outputs, val_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc = torch.eq(predict_y, val_labels).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                if total_test_step % 10 == 0:
                    writer.add_scalar("test_loss", loss.item(), total_test_step)
                    writer.add_scalar("test_accuracy", acc / batch_size, total_test_step)
                total_test_step += 1

        val_accurate = acc / len(test_dataloader)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(resnet.state_dict(), save_path)

    print('Finished Training')
    torch.save(resnet.state_dict(), "model.pth")


if __name__ == '__main__':
    main()
    writer.close()
    notify.note()