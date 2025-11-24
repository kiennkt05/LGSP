import torch  
import torch.nn as nn  
import torch.nn.functional as F  

from utils import *
from timm.models import create_model
from models.rainbow.prompt import RainbowPromptModule


class PrefixAttention(nn.Module):
    """Attention wrapper that supports prefix prompts for Rainbow."""
    
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        
    def forward(self, x, prompt=None):
        B, N, C = x.shape
        
        # Get QKV from original attention
        qkv = self.original_attn.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        if prompt is not None:
            # prompt shape: [B, 2, length, num_heads, head_dim]
            # Convert to [2, B, num_heads, length, head_dim]
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()
            key_prefix = prompt[0]  # [B, num_heads, length, head_dim]
            value_prefix = prompt[1]  # [B, num_heads, length, head_dim]
            
            # Concatenate prefix to keys and values
            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.original_attn.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.original_attn.proj(x)
        x = self.original_attn.proj_drop(x)
        return x


class BlockWrapper(nn.Module):
    """Wrapper for timm Block to support prefix prompts."""
    
    def __init__(self, original_block):
        super().__init__()
        self.block = original_block
        # Store original attention
        self.original_attn = original_block.attn
        # Replace attention with prefix-aware version
        self.block.attn = PrefixAttention(original_block.attn)
        
    def forward(self, x, prompt=None):
        # Apply norm1
        x_norm = self.block.norm1(x)
        # Apply attention with prompt
        attn_out = self.block.attn(x_norm, prompt=prompt)
        # Apply layer scale and drop path if exists
        if hasattr(self.block, 'ls1'):
            attn_out = self.block.ls1(attn_out)
        if hasattr(self.block, 'drop_path1'):
            attn_out = self.block.drop_path1(attn_out)
        x = x + attn_out
        
        # MLP branch
        mlp_out = self.block.mlp(self.block.norm2(x))
        if hasattr(self.block, 'ls2'):
            mlp_out = self.block.ls2(mlp_out)
        if hasattr(self.block, 'drop_path2'):
            mlp_out = self.block.drop_path2(mlp_out)
        x = x + mlp_out
        return x


class ViT_Rainbow(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        
        # Set default num_features (768 for ViT-Base)
        # This will be used if dataset is not in the specific list
        self.num_features = 768
        
        if self.args.dataset in ['cifar100', 'cub200', 'mini_imagenet', 'FGVCAircraft', 'iNF200', 'air']:
            self.num_features = 768
        
        self.encoder = create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=args.num_classes,
                                drop_rate=0., drop_path_rate=0., drop_block_rate=None)
        
        # Get num_heads before wrapping blocks
        num_heads = self.encoder.blocks[0].attn.num_heads
        
        # Wrap blocks to support prefix prompts
        self.encoder.blocks = nn.ModuleList([BlockWrapper(block) for block in self.encoder.blocks])
        
        # Classifier Head as a Fully Connected Layer
        self.classifier_head = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        
        self.seen_classes = args.base_class
        self.way = args.way
        self.base_class = args.base_class
        
        # Initialize RainbowPromptModule
        embed_dim = self.encoder.embed_dim
        num_layers = len(self.encoder.blocks)
        # num_heads was captured before wrapping blocks
        
        # Get Rainbow config from args
        prompt_length = getattr(args, 'rainbow_prompt_length', 5)
        proj_dim = getattr(args, 'rainbow_proj_dim', embed_dim // 8)
        align_hidden_dim = getattr(args, 'rainbow_align_hidden_dim', embed_dim // 8)
        gate_tau_start = getattr(args, 'rainbow_gate_tau_start', 1.0)
        gate_tau_end = getattr(args, 'rainbow_gate_tau_end', 0.3)
        gate_harden_at = getattr(args, 'rainbow_gate_harden_at', 0.6)
        save_dir = getattr(args, 'rainbow_save_dir', './checkpoint/rainbow_prompts')
        use_paper_evolution = getattr(args, 'rainbow_use_paper_evolution', False)
        
        self.rainbow_prompt = RainbowPromptModule(
            embed_dim=embed_dim,
            prompt_length=prompt_length,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_dim=proj_dim,
            align_hidden_dim=align_hidden_dim,
            gate_tau_start=gate_tau_start,
            gate_tau_end=gate_tau_end,
            gate_harden_at=gate_harden_at,
            save_dir=save_dir,
            use_task_conditioning=True,
            enable_task_level=True,
            enable_feature_level=True,
            enable_alignment=True,
            use_adaptive_gating=True,
            use_paper_evolution=use_paper_evolution,
        )
        
        self.lambda_sparse = getattr(args, 'rainbow_lambda_sparse', 0.0)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.seen_classes += len(new_classes)
    
    def encode(self, x):
        x = self.encoder.forward_features(x)[:,0]
        return x
    
    def prompt_encode(self, img, task_id=-1, train=True):
        x = self.encoder.patch_embed(img)  # (batch_size, 196, embed_dim)
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([ex_cls, x], dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        
        # Set training mode for Rainbow
        self.rainbow_prompt.set_training(train)
        
        # Process through blocks with Rainbow prompts
        for i, block in enumerate(self.encoder.blocks):
            prompt_tokens = self.rainbow_prompt(
                task_id=task_id,
                layer_idx=i,
                batch_size=x.shape[0],
                device=x.device,
            )
            x = block(x, prompt=prompt_tokens)
        
        x = self.encoder.norm(x)
        x = x[:, 0, :]
        return x

    def forward(self, input, query=False, memory_data=None, session=-1):
        res = {}
        
        if query:
            q_feat = self.encode(input)
            return q_feat

        # Use session as task_id for Rainbow
        task_id = session if session >= 0 else 0
        train = self.training
        embedding = self.prompt_encode(input, task_id=task_id, train=train)
        logit = self.classifier_head(embedding)

        res['logit'] = logit
        
        # Get auxiliary losses from Rainbow
        aux_losses = self.rainbow_prompt.auxiliary_losses()
        if aux_losses:
            res['rainbow_aux'] = aux_losses

        if memory_data is not None:
            res['logit'] = torch.cat([logit, memory_data], dim=0)
        return res

    def train_inc(self, dataloader, epochs, session, class_list, testloader, result_list, test, model_test):
        print("[Session: {}]".format(session))
        self.update_fc_avg(dataloader, class_list)
        optimizer_params = []

        # Rainbow prompt parameters
        for layer_idx in range(len(self.encoder.blocks)):
            for prompt in self.rainbow_prompt.base_prompts[layer_idx]:
                if prompt.requires_grad:
                    optimizer_params.append({'params': [prompt], 'lr': self.args.lr_new})
        
        # Rainbow evolution parameters
        optimizer_params.append({'params': self.rainbow_prompt.evolutions.parameters(), 'lr': self.args.lr_new})
        
        # Rainbow gate parameters
        if self.rainbow_prompt.current_gate is not None:
            optimizer_params.append({'params': self.rainbow_prompt.current_gate.parameters(), 'lr': self.args.lr_new})

        # Classifier
        params_classifier = [p for p in self.classifier_head.parameters()]
        optimizer_params.append({'params': params_classifier, 'lr': self.args.lr_new})

        optim = torch.optim.Adam(optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * 1)
        
        best_epoch = -1
        best_accuracy = 0.0
        last_novel_acc = 0.0
        final_tsa = 0.0  # Initialize to avoid UnboundLocalError if epochs=0

        for epoch in range(epochs):
            # Set epoch for Rainbow
            self.rainbow_prompt.set_epoch(epoch, epochs)
            
            # Accumulate metrics across batches
            tl = Averager_Loss()
            ta = Averager()
            
            for idx, batch in enumerate(dataloader):
                data_imgs, data_label = [_.cuda() for _ in batch]

                self.train()

                res = self.forward(data_imgs, memory_data=None, session=session)
                logits = res['logit']

                seen_class = self.base_class + session * self.way
                logits = logits[:, :seen_class]

                loss_ce = F.cross_entropy(logits, data_label)
                
                # Add Rainbow auxiliary losses
                loss = loss_ce
                if 'rainbow_aux' in res:
                    aux_losses = res['rainbow_aux']
                    sparsity_loss = sum(aux_losses.values()) if aux_losses else torch.tensor(0.0, device=loss_ce.device)
                    loss = loss_ce + self.lambda_sparse * sparsity_loss
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                pred = torch.argmax(logits, dim=1)
                acc = (pred == data_label).sum().item() / data_label.shape[0] * 100.
                
                # Accumulate metrics
                tl.add(loss.item(), len(data_label))
                ta.add(acc, len(data_label))
            
            # Step scheduler once per epoch, not per batch
            scheduler.step()
            lrc = scheduler.get_last_lr()[0]
            tsl, tsa, logs = test(model_test, testloader, self.args, session)
            # Keep last_novel_acc as ratio (0-1) to match final_tsa units
            # The trainer will handle conversion to percentage if needed
            last_novel_acc = logs.get('new_acc', 0.0)
            if tsa > best_accuracy:
                best_accuracy = tsa
                best_epoch = epoch

            # Get averaged metrics
            avg_loss = tl.item()
            avg_acc = ta.item()

            result_list.append(
                'epoch:%03d,lr:%.4f,B:%.5f,N:%.5f,BN:%.5f,NB:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                    epoch, lrc, logs['base_acc'], logs['new_acc'], logs['base_acc_given_new'], logs['new_acc_given_base'], avg_loss, avg_acc, tsl, tsa
                )
            )
            # Store final test accuracy for return (from last epoch)
            final_tsa = tsa
            
        result_list.append('Session {}, Best test_Epoch {}, Best test_Acc {:.4f}'.format(
            session, best_epoch, best_accuracy))

        return final_tsa, last_novel_acc
    
    def update_fc_avg(self, dataloader, class_list):
        self.eval()
        query_p = []
        
        # Accumulate embeddings and labels across all batches
        embedding_list = []
        label_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                cls_embed = self.encode(data_imgs).detach()
                embedding_list.append(cls_embed.cpu())
                label_list.append(label.cpu())
        
        # Concatenate all embeddings and labels
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        with torch.no_grad():
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                if len(data_index) > 0:
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    query_p.append(proto)
                    self.classifier_head.weight.data[class_index] = proto.to(self.classifier_head.weight.device)
        
        if query_p:
            query_p = torch.stack(query_p)
        
        self.train()

