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
        wrapped_blocks = nn.ModuleList([BlockWrapper(block) for block in self.encoder.blocks])
        self.encoder.blocks = wrapped_blocks
        
        # Override forward_features to handle wrapped blocks
        # Capture encoder reference in closure
        encoder_ref = self.encoder
        def custom_forward_features(x):
            x = encoder_ref.patch_embed(x)
            x = torch.cat([encoder_ref.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = encoder_ref.pos_drop(x + encoder_ref.pos_embed)
            # Manually iterate through wrapped blocks (without prompts for query mode)
            for block in encoder_ref.blocks:
                x = block(x, prompt=None)
            x = encoder_ref.norm(x)
            return x
        
        self.encoder.forward_features = custom_forward_features
        
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
        
        # Pixel prompt initialization
        self.prompt_dropout = torch.nn.Dropout(getattr(args, 'Dropout_Prompt', 0.0))
        self.first_kernel_size = getattr(args, 'first_kernel_size', 3)
        self.second_kernel_size = getattr(args, 'second_kernel_size', 3)
        
        def build_prompt_module():
            prompt_hid_dim = getattr(args, 'prompt_hid_dim', 64)
            return nn.Sequential(
                nn.Conv2d(3, prompt_hid_dim, self.first_kernel_size, stride=1, padding=int((self.first_kernel_size - 1) / 2)),
                nn.ReLU(),
                nn.Conv2d(prompt_hid_dim, 3, self.second_kernel_size, stride=1, padding=int((self.second_kernel_size - 1) / 2))
            )

        self.prompt_generators = nn.ModuleList()

        if getattr(args, 'pixel_prompt', 'NO') == "YES":
            pool_size = getattr(args, 'pool_size', 10)
            self.prompt_generators = nn.ModuleList(
                build_prompt_module() for _ in range(pool_size)
            )
            self.num_prompt_generators = pool_size
        else:
            self.num_prompt_generators = 0

        # Frequency mask initialization
        if getattr(args, 'Frequency_mask', False):
            max_radius = torch.sqrt(torch.tensor((224 / 2) ** 2 + (224 / 2) ** 2)).item()
            num_r = getattr(args, 'num_r', 10)
            self.radii = torch.linspace(0, max_radius, steps=num_r)
            weights_init = torch.normal(mean=0, std=10, size=(num_r,))
            self.weights = nn.Parameter(weights_init)

            if getattr(args, 'adaptive_weighting', False):
                self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
                self.beta = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.seen_classes += len(new_classes)
    
    # Define a function to calculate cosine similarity, focusing more on local similarity.
    def cosine_similarity(self, a, b):
        # Normalize the vectors
        a_norm = F.normalize(a, dim=1)  # [batch_size, channels, h, w]
        b_norm = F.normalize(b, dim=1)  # [batch_size, channels, h, w]
        # Calculate the dot product
        return torch.sum(a_norm * b_norm, dim=1, keepdim=True)  # [batch_size, 1, h, w]
    
    def get_prompts(self, x, session=-1):
        res = {}  # Initialize res as an empty dictionary
        prompts_list = []
        for prompt_net in self.prompt_generators:
            prompts_list.append(self.prompt_dropout(prompt_net(x)))

        self.num_prompt_generators = len(prompts_list)

        # Use point-wise convolution to increase the channel dimension
        # Feature normalization, such as BatchNorm or GroupNorm, to reduce redundant information
        # Perform softmax on the branches
        similarities_list = [self.cosine_similarity(x, prompt) for prompt in prompts_list]  # Each element is [batch_size, 1, h, w]

        # Concatenate all similarities and perform softmax normalization
        similarities = torch.cat(similarities_list, dim=1)  # [batch_size, 20, h, w]
        weights = F.softmax(similarities, dim=1)  # [batch_size, 20, h, w]
        
        # Stack prompts and perform weighted sum
        prompts = torch.stack(prompts_list, dim=1)  # [batch_size, 10, channels, h, w]
        weighted_prompt = torch.sum(weights.unsqueeze(2) * prompts, dim=1)  # [batch_size, channels, h, w]
        prompts = weighted_prompt

        res['prompts'] = prompts
        return res
    
    def get_Frequency_mask(self, input):
        # Perform Fourier transform on the h and w dimensions
        fft_im = torch.fft.fftn(input, dim=(-2, -1))  # 2D Fourier transform
        fft_im_center = torch.fft.fftshift(fft_im, dim=(-2, -1))  # Shift the zero frequency to the center

        # Build a grid to calculate the distance from each point to the center of the spectrum
        Batch_size, channels, h, w = input.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2  # Center of the spectrum
        distances = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)  # Distance matrix
        distances = distances.to(input.device)  # Ensure the device is consistent

        # Create a ring mask, allow a certain tolerance range
        beta = 4.0
        ring_masks = []  # Store the mask of each ring
        for i, radius in enumerate(self.radii):
            if i == 0:
                inner_radius = 0  # The first ring starts from the center
            else:
                inner_radius = self.radii[i - 1] + 1e-6  # Ensure no overlap

            # Outer radius mask
            outer_mask = torch.sigmoid(-beta * (distances - radius))
            # Inner radius mask
            inner_mask = torch.sigmoid(-beta * (distances - inner_radius))
            # Ring mask
            ring_mask = outer_mask - inner_mask
            ring_masks.append(ring_mask.float())  # Convert to float, for subsequent operations

        # Stack the masks into a tensor of shape [10, h, w]
        ring_masks = torch.stack(ring_masks, dim=0).to(input.device)  # [10, h, w]

        # Weight each ring
        temperature = getattr(self.args, 'temperature', 1.0)
        weights_normalized = torch.softmax(self.weights * temperature, dim=0)  # Normalize the weights
        weighted_ring_masks = weights_normalized[:, None, None] * ring_masks  # Weighted mask
        # Sum the weighted masks, get the overall frequency mask
        final_mask = weighted_ring_masks.sum(dim=0)  # [h, w]

        # Apply the frequency mask
        fft_selected = fft_im_center * final_mask[None, None, :, :]  # Broadcast to [Batch_size, 3, h, w]

        # Use the residual operation
        fft_residual = fft_im_center + fft_selected  # Original frequency + weighted ring
        ifft_residual = torch.fft.ifftn(torch.fft.ifftshift(fft_residual, dim=(-2, -1)), dim=(-2, -1))
        ifft_residual = torch.abs(ifft_residual)  # [Batch_size, 3, h, w]
        output = input + (ifft_residual - input) * 0.1
        return output
    
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
        
        # Apply pixel_prompt and frequency_mask preprocessing
        pixel_prompt_enabled = getattr(self.args, 'pixel_prompt', 'NO') == 'YES'
        frequency_mask_enabled = getattr(self.args, 'Frequency_mask', False)
        adaptive_weighting = getattr(self.args, 'adaptive_weighting', False)
        
        if adaptive_weighting:
            input1 = None
            input2 = None
            if pixel_prompt_enabled:
                res = self.get_prompts(input, session=session)  
                prompts = res['prompts']
                input1 = input + prompts * 1
            if frequency_mask_enabled:
                input2 = self.get_Frequency_mask(input)
            
            if input1 is not None and input2 is not None:
                input = self.alpha * input1 + self.beta * input2
            elif input1 is not None:
                input = input1
            elif input2 is not None:
                input = input2
        else:
            if pixel_prompt_enabled:
                res = self.get_prompts(input, session=session)  
                prompts = res['prompts']
                input = input + prompts * 1
            if frequency_mask_enabled:
                input = self.get_Frequency_mask(input)
        
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

        # Frequency mask parameters (if enabled)
        if getattr(self.args, 'Frequency_mask', False):
            params_Frequency_mask = [self.weights]
            optimizer_params.append({'params': params_Frequency_mask, 'lr': getattr(self.args, 'lr_Frequency_mask', 0.03) * 0.05})

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