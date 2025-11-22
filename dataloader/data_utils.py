import numpy as np
import torch

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'FGVCAircraft':  
        import dataloader.FGVCAircraft.FGVCAircraft as Dataset
        args.base_class = 50
        args.num_classes = 100  
        args.way = 5
        args.shot = 5
        args.sessions = 11  
    if args.dataset == 'iNF200':  
        import dataloader.iNF200.iNF200 as Dataset
        args.base_class = 100
        args.num_classes = 200 
        args.way = 10
        args.shot = 5
        args.sessions = 11 
    args.Dataset = Dataset
    return args

def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args, clip_trsf=None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, transform=clip_trsf,
                                        index=class_index, base_session=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_session=True)
    elif args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, base_session=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)
    elif args.dataset == 'FGVCAircraft':
        trainset = args.Dataset.FGVCAircraft(root=args.dataroot, train=True, index_path=txt_path, base_session=True)
        testset = args.Dataset.FGVCAircraft(root=args.dataroot, train=False, index=class_index)
    elif args.dataset == 'iNF200':
        trainset = args.Dataset.iNF200(root=args.dataroot, train=True, index_path=txt_path, base_session=True)
        testset = args.Dataset.iNF200(root=args.dataroot, train=False, index=class_index)
    elif args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_session=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)
    
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=args.batch_size_base, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    return trainset, trainloader, testloader

def get_new_dataloader(args, session, clip_trsf=None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                        index=class_index, base_session=False)
    elif args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(
            root=args.dataroot, train=True, index_path=txt_path)
    elif args.dataset == 'FGVCAircraft':
        trainset = args.Dataset.FGVCAircraft(
            root=args.dataroot, train=True, index_path=txt_path)
    elif args.dataset == 'iNF200':
        trainset = args.Dataset.iNF200(
            root=args.dataroot, train=True, index_path=txt_path)
    elif args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=True, index_path=txt_path)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset, batch_size=batch_size_new, shuffle=False, num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False, transform=clip_trsf,
                                        index=class_new, base_session=False)
    elif args.dataset == 'cub200':
        testset = args.Dataset.CUB200(
            root=args.dataroot, train=False, index=class_new)
    elif args.dataset == 'FGVCAircraft':
        testset = args.Dataset.FGVCAircraft(
            root=args.dataroot, train=False, index=class_new)
    elif args.dataset == 'iNF200':
        testset = args.Dataset.iNF200(
            root=args.dataroot, train=False, index=class_new)
    elif args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=False, index=class_new)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list