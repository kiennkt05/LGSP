from .base import Trainer
import torch
import pandas as pd
import time

from .helper import *
from utils import *
from dataloader.data_utils import *
from models.base.ViT_Network import ViT_MYNET

class ViT_FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_save_path()
        self.set_log_path()

        self.model = ViT_MYNET(self.args, mode=self.args.base_mode)
        self.model = self.model.to(self.device)
        
        # Freeze the encoder parameters
        for p in self.model.encoder.parameters():
            p.requires_grad=False
        
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            # self.best_model_dict = deepcopy(self.model.state_dict())
    
        print("#"*50)
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        self.init_params = sum(param.numel() for param in self.model.parameters())
        print('total parameters:',self.init_params)
        print('trainable parameters:',trainable_params)
        print("#"*50)
        # self.writer = SummaryWriter(f'runs/all_phases_val/epoch{self.args.epochs_base}_{self.args.epochs_new}')

    def get_optimizer_base(self):
        optimizer_params = []
        
        if self.args.pixel_prompt == "YES":
            prompt_branch_params = []
            for prompt_net in self.model.prompt_generators:
                prompt_branch_params.extend(list(prompt_net.parameters()))

            params_Mask = [
                p
                for p in prompt_branch_params
                if p.requires_grad
            ]
            optimizer_params.append({'params': params_Mask, 'lr': self.args.lr_local})
  
        if self.args.Frequency_mask:
            params_Frequency_mask = [self.model.weights]
            optimizer_params.append({'params': params_Frequency_mask, 'lr': self.args.lr_Frequency_mask})

        if self.args.adaptive_weighting:
            params_test1 = [self.model.alpha, self.model.beta]
            optimizer_params.append({'params': params_test1, 'lr': 0.1})
        # VPT
        params_vpt = [self.model.Prompt_Tokens]
        optimizer_params.append({'params': params_vpt, 'lr': self.args.lr_PromptTokens_base})

  
        params_classsifier = [p for p in self.model.classifier_head.parameters()]
        optimizer_params.append({'params': params_classsifier, 'lr': self.args.lr_base})

        optimizer = torch.optim.Adam(optimizer_params)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            return get_base_dataloader(self.args)
        return get_new_dataloader(self.args, session)

    def train(self):
        args = self.args
        t_start_time = time.time()
        result_list = [args]
        columns = ['num_session', 'acc', 'base_acc', 'new_acc', 'base_acc_given_new', 'new_acc_given_base']
        acc_df = pd.DataFrame(columns=columns)
        print("[Start Session: {}] [Sessions: {}]".format(args.start_session, args.sessions))
        
        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            print(f"Session: {session} Data Config")
            print(len(train_set.targets))
            if session == 0:
                if self.args.pixel_prompt == "YES":
                    checkpoint_path = "run_script/meta_net_2_params_lastBaseEpoch.pth"
                    state_dict = torch.load(checkpoint_path, map_location=self.device)

                    for prompt_net in self.model.prompt_generators:
                        prompt_net.load_state_dict(state_dict)
            if session > 0:
                if self.args.pixel_prompt == "YES":
                    for prompt_net in self.model.prompt_generators:
                        for p in prompt_net.parameters():
                            p.requires_grad = False

                if self.args.Frequency_mask:
                    self.model.weights.requires_grad = True
                    # for p in self.model.AdaptiveFrequencyMask.parameters():
                    #     p.requires_grad = False

            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                
                # build_base_proto(trainloader, self.model, self.query_info, args)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
            
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    train_loss, train_acc = base_train(self.model, trainloader, optimizer, scheduler, epoch, np.unique(train_set.targets), args)
                    test_loss, test_acc, logs = test(self.model, testloader, args, session)

                    # self.writer.add_scalar('Accuracy/Base_Phase', tsa, epoch)

                    if (test_acc * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (test_acc * 100))
                        self.trlog['max_acc_epoch'][session] = epoch
                        print('********A better model is found!!**********')
                        # print('Saving model to :%s' % save_model_dir)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,B:%.5f,N:%.5f,BN:%.5f,NB:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, logs['base_acc'], logs['new_acc'], logs['base_acc_given_new'], logs['new_acc_given_base'], train_loss, train_acc*100, test_loss, test_acc*100
                        )
                    )
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                            '\nstill need around %.2f mins to finish this session' % (
                                    (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'][session], self.trlog['max_acc'][session], ))

                #*=======================================================================================
                if not args.not_data_init:
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.model.mode = 'avg_cos'
                    test_loss, test_acc, logs = test(self.model, testloader, args, session)
                    self.trlog['max_acc'][session] = float('%.3f' % (test_acc * 100))
                    result_list.append('After Prototype FC: test_loss:%.5f,test_acc:%.5f\n' % (test_loss, test_acc))
            else:  # incremental learning sessions
                print("Incremental session: [%d]" % session)
                print("#"*50)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
            
                self.model.update_seen_classes(np.unique(train_set.targets))

                self.model.mode = self.args.new_mode
                self.model.train()
                trainloader.dataset.transform = testloader.dataset.transform
                # self.model.module.train_inc(trainloader, self.args.epochs_new, session, np.unique(train_set.targets), self.word_info, self.query_info)
                test_acc, novel_last_acc = self.model.train_inc(trainloader, self.args.epochs_new, session, np.unique(train_set.targets), testloader, result_list, test, self.model)
                
                self.trlog['max_acc'][session] = float('%.3f' % (test_acc * 100))
                novel_idx = session - 1
                if 0 <= novel_idx < len(self.trlog['novel_acc']):
                    self.trlog['novel_acc'][novel_idx] = float('%.3f' % novel_last_acc)
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
        result_list.append(self.trlog['max_acc'])

        print("#"*50)
        print("#"*17 + "END OF TRAINING" + "#"*18)
        print("#"*50)

        novel_sessions = [s for s in range(self.args.start_session, self.args.sessions) if s > 0]
        novel_last_epoch_acc = [self.trlog['novel_acc'][s - 1] for s in novel_sessions]

        print('Incremental Novel last-epoch accuracy (%):\n', novel_last_epoch_acc)
        print('Last session test accuracy:\n', self.trlog['max_acc'])
        if self.args.adaptive_weighting:
            print('Adaptive weights: (alpha, beta) = ({}, {})'.format(self.model.alpha, self.model.beta))
        print()
        max_acc = self.trlog['max_acc']
        first_value = max_acc[0]
        average = sum(max_acc) / len(max_acc)
        if novel_last_epoch_acc:
            novel_avg = sum(novel_last_epoch_acc) / len(novel_last_epoch_acc)
        print("#"*50)
        print("#"*18 + "Final Results" + "#"*19)
        print("#"*50)
        print("#"*12 + f"O: {average:.2f} B: {first_value:.2f} N: {novel_avg:.2f}" + "#"*12)
        print("#"*50)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'][0])
        print('Total time used %.2f mins' % total_time)
        
        end_params = sum(param.numel() for param in self.model.parameters())
        print('[Begin] Total parameters: {}'.format(self.init_params))
        print('[END] Total parameters: {}'.format(end_params))
        
    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/%s/' % (self.args.dataset, time.strftime("%Y%m%d_%H%M%S")) + 'base_ViT_Ours/'

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.seed)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.seed)
        else:
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-COS_%d-Gam_%.2f-Bs_%d-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.seed)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        self.args.save_path = os.path.join(f'checkpoint/{self.args.out}', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

    def set_log_path(self):
        if self.args.model_dir is not None:
            self.args.save_log_path = 'base/' + '%s' % self.args.dataset
            if 'avg' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_prototype_' + self.args.model_dir.split('/')[-2][:7] + '/'
            if 'ft' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_WaRP_' + 'lr_new_%.3f-epochs_new_%d-keep_frac_%.2f/' % (
                    self.args.lr_new, self.args.epochs_new, self.args.fraction_to_keep)
            self.args.save_log_path = os.path.join('acc_logs', self.args.save_log_path)
            ensure_path(self.args.save_log_path)
            self.args.save_log_path = self.args.save_log_path + self.args.model_dir.split('/')[-2] + '.csv'

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)