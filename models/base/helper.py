import math
from utils import *
from tqdm import tqdm
import torch.nn.functional as F  
import torch  

def replace_base_fc(trainset, transform, model, args):
    print("[Replace Base FC - Original]")
    model = model.eval()

    base_batch_size = getattr(args, "replace_fc_batch_size", 128)
    num_generators = max(1, getattr(model, "num_prompt_generators", 1))
    reference_generators = getattr(args, "replace_fc_reference_generators", 30)
    scale_factor = max(1, math.ceil(num_generators / max(1, reference_generators)))
    adaptive_batch_size = max(1, base_batch_size // scale_factor)

    if adaptive_batch_size < base_batch_size:
        print(f"[Replace Base FC] Adaptive batch size {adaptive_batch_size} (scale_factor={scale_factor}) to reduce memory usage.")

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=adaptive_batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            device = next(model.parameters()).device
            data, label = [_.to(device) for _ in batch]
            # model.module.mode = 'encoder'
            model.mode = 'encoder'
            embedding = model(data, query=True)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
        
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    # model.module.fc.weight.data[:args.base_class] = proto_list
    model.classifier_head.weight.data[:args.base_class] = proto_list

    return model

def cross_entropy(preds, targets, reduction='none'):
    device = targets.device
    labels = torch.arange(targets.shape[0], device=device)
    loss = F.cross_entropy(preds,labels, reduction='none')
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
# The scheduler is a learning rate regulator; I used to manually adjust the learning rate throughout my graduation project.
def base_train(model, trainloader, optimizer, scheduler, epoch, class_list, args):
    print("[Base Train]")
    # base_mode = model.module.mode
    base_mode = model.mode
    
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader, mininterval=1.0)

    model.mode = "encoder"
  
    for i, batch in enumerate(tqdm_gen, 1):
        device = next(model.parameters()).device
        data, train_label = [_.to(device) for _ in batch]

        res = model(data)
        logits = res['logit']
        # logits = model(data, base=False, epoch=epoch, batch_num=i)

        logits_ = logits[:, :args.base_class]

        # T = 0.5

        loss_ce = F.cross_entropy(logits_, train_label)

        # loss_ce = F.cross_entropy(logits_ / T, train_label)

        # loss_ce = F.cross_entropy(logits_ / T, train_label) * (T ** 2)        
        acc = count_acc(logits_, train_label)
        # total_loss = loss_ce + args.ED_hp*loss_tri + loss_kb
        total_loss = loss_ce

        # # 打印两个损失的值范围
        # print(f"Total loss (before scaling): {total_loss.item()}")
        # print(f"Reduce_sim: {res['reduce_sim'].item()}")
        
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        ta.add(acc, len(train_label))
        
        # tqdm_gen.set_description(
        #     'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, loss_CE={:.4f}, loss_ED={:.4f}, loss_SKD={:.4f}, acc={:.4f}'.\
        #         format(epoch, lrc, total_loss.item(), loss_ce.item(), loss_tri.item(), loss_kb.item(), ta.item()))
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, loss_CE={:.4f}, acc={:.4f}'.\
                format(epoch, lrc, total_loss.item(), loss_ce.item(), ta.item() * 100))
        
        optimizer.zero_grad()
        total_loss.backward()
        
        optimizer.step()
        
    tl = tl.item()
    ta = ta.item()
    
    model.mode = base_mode
    return tl, ta

def test(model, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
    # print("\t\t\t[Test Phase] Session: {}".format(session))
    with torch.no_grad():
        # 取消可视化
        tqdm_gen = tqdm(testloader, disable=True)
        # tqdm_gen = tqdm(testloader, disable=False)
        for i, batch in enumerate(tqdm_gen, 1):
            device = next(model.parameters()).device
            data, test_label = [_.to(device) for _ in batch]

            res = model(data)
            logits = res['logit']

            # logits = model(data)
        
            logits = logits[:, :test_class]
            
            loss = F.cross_entropy(logits, test_label)
            
            acc = count_acc(logits, test_label)

            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs])
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs])
                va_base.add(acc_base, len(test_label[base_idxs]))
                va_base_given_new.add(acc_base_given_new, len(test_label[base_idxs]))

            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class)
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs])
                va_new.add(acc_new, len(test_label[new_idxs]))
                va_new_given_base.add(acc_new_given_base, len(test_label[new_idxs]))

            vl.add(loss.item(), len(test_label))
            va.add(acc, len(test_label))

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()

    # print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch,))
    # print()
    # print('BN={:.4f}, NB={:.4f}, l={:.4f}, B={:.4f}, N={:.4f}, na={:.4f}, ba={:.4f}'.format(va_base_given_new*100, va_new_given_base*100, vl, va_base*100, va_new*100, va*100, base_acc))
    # print()
    print('BN={:.4f}, NB={:.4f}, B={:.4f}, N={:.4f}, l={:.4f}, acc={:.4f},'.format(va_base_given_new*100, va_new_given_base*100, va_base*100, va_new*100, vl, va*100))
    # print('base acc given new : {:.4f}'.format(va_base_given_new))
    # print('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)
    return vl, va, logs

def build_base_proto(train_loader, model, query_info, args):
    model = model.eval()
    
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            device = next(model.parameters()).device
            data, label = [_.to(device) for _ in batch]
            
            # model.module.mode = 'encoder'
            model.mode = 'encoder'
            embedding = model(data, query=True)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
            
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        # tensor([[0], [3]])
        # 通过.squeeze(-1)得到tensor([0，3])
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0) #* num_base, feat_dim
    query_info["proto"] = proto_list
    # model.module.mode = args.base_mode
    model.mode = args.base_mode
    model = model.train()