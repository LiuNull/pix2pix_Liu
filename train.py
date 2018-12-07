### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import pynvml
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
from util import html
from util import fid_score


def validate(model,test_dataset,visualizer,epoch,total_steps,epoch_iter):
    test_web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%d' % (opt.phase, epoch,epoch_iter))
    test_webpage = html.HTML(test_web_dir, 'Experiment = %s, Phase = %s, Epoch = %s, Iter = %d' % (opt.name, opt.phase, epoch, epoch_iter))
    if total_steps % 200 ==0:
        torch.set_grad_enabled(False)
        model.eval()
        # generate the test images for calculating FID
        for test_i, test_data in enumerate(test_dataset):
            _, test_generated = model(Variable(test_data['label']), Variable(test_data['inst']), Variable(test_data['image']), Variable(test_data['feat']),Variable(test_data['labelTrain']), infer=True)
            test_visuals = OrderedDict([('input_label', util.tensor2label(test_data['label'][0], opt.label_nc)),
                            ('synthesized_image', util.tensor2im(test_generated.data[0]))])
            test_img_path = test_data['path']
            # print('process image... %s' % test_img_path)
            visualizer.save_images(test_webpage, test_visuals, test_img_path,epoch_iter)
            visualizer.display_current_results(test_visuals, epoch, total_steps)

        # calculate the FID for test_data
        test_path = os.path.join(test_web_dir,'images')
        compare_path = '/home/hsx/Datasets/cityscapes/leftImg8bit/val/'
        paths = []
        paths.append(test_path)
        paths.append(compare_path)
        fid_value = fid_score.calculate_fid_given_paths(paths,batch_size=64,dims=2048,cuda=False)
        message = 'Epoch:%d, iteration:%d, the test fid: %f' %(epoch,epoch_iter,fid_value)
        print(message)
        testlog_name = os.path.join(opt.checkpoints_dir, opt.name, 'test_fid_log.txt')
        with open(testlog_name, "a") as log_file:
            log_file.write('%s\n' % message)


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

opt.phase = 'test'
opt.isTrain = False
test_data_loader = CreateDataLoader(opt)
test_dataset = test_data_loader.load_data()
test_dataset_size = len(test_data_loader)
print('#Test images = %d' % test_dataset_size)
opt.phase = 'train'
opt.isTrain = True

model = create_model(opt)
print("successfully loaded model")
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        validate(model,test_dataset,visualizer,epoch,total_steps,epoch_iter)
        torch.set_grad_enabled(True)
        model.train()
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        labelTrain = Variable(data['labelTrain'])
        # print(labelTrain[0,0].shape) torch.Size([512, 1024])
        # np.savetxt("labelTrain.txt",labelTrain[0,0].numpy())

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']),Variable(data['labelTrain']), infer=save_fake)

        # print(ERF_fake_loss) tensor(3.2985, device='cuda:1')

        # sum per device losses
        # if multi-gpu, need the following code
        # losses = [ torch.mean(x) if not isinstance(x, int)  else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + loss_dict['ERF_fake']
        # loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + loss_dict['ERF_fake']
        # loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict['ERF_fake']
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict['ERF_fake']


        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()
        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()
        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 
        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            # errors.update('5'=ERF_fake_loss)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()