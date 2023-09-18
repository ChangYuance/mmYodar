import os

import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_lr
batch_size = 16

def fit_one_epoch(model1_train, model1, model2_train, model2, yolo_loss, loss_history1, loss_history2, eval_callback, optimizer1,optimizer2,epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    print('Start Train')
    pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model1_train.train()
    model2_train.train()
    mse = torch.nn.MSELoss()
    for iteration, (input1, input2, targets) in enumerate(gen):
        if iteration >= epoch_step:
            break

        with torch.no_grad():
            if cuda:
                #input1    = input1.cuda()
                input2    = input2.cuda()
                targets  = [ann.cuda() for ann in targets]

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        if not fp16:

            start_time = time.time()
            print(input2)
            plt.imshow(input2)
            plt.show()
            outputs1         = model1_train(input2)
            end_time = time.time()
            print(end_time-start_time)

            loss_value_all  = 0

            for l in range(len(outputs1)-3):
                loss_value_all  += yolo_loss(l, outputs1[l], targets)
            loss_value1 = loss_value_all
            

            loss_value1.backward()
            optimizer1.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():


                outputs1         = model1_train(input2)
                outputs2         = model2_train(input2)
                loss_value_all  = 0

                for l in range(len(outputs1)-3):
                    loss_value_all  += yolo_loss(l, outputs1[l], targets)
                loss_value1 = loss_value_all
                loss_value_all  = 0
                for l in range(len(outputs2)-3):
                    loss_value_all  += yolo_loss(l, outputs2[l], targets)
                loss_value2 = loss_value_all

                temp = 0
                for l in range(len(outputs1)-3):
                    temp += mse(outputs1[l], outputs2[l])
                loss_value1 += temp
                loss_value2 += temp
            scaler.scale(loss_value1).backward(retain_graph=True)
            scaler.scale(loss_value2).backward(retain_graph=True)
            scaler.step(optimizer1)
            scaler.step(optimizer2)
            scaler.update()

        loss += loss_value1.item()
        
    pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer1)})
    pbar.update(1)

    pbar.close()
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    print("model1")
    model1_train.eval()
    for iteration, (input1, input2, targets) in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        with torch.no_grad():
            if cuda:
                input1  = input1.cuda()
                targets = [ann.cuda() for ann in targets]

            optimizer1.zero_grad()

            outputs         = model1_train(input1)

            loss_value_all  = 0

            for l in range(len(outputs)-3):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
            loss_value  = loss_value_all

        val_loss += loss_value.item()
    pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
    pbar.update(1)
 
    if 1:
        pbar.close()
        print('Finish Validation')
        loss_history1.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model1_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model1.state_dict(), os.path.join(save_dir, "model1_ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history1.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history1.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model1.state_dict(), os.path.join(save_dir, "model1_best_epoch_weights.pth"))
            
        torch.save(model1.state_dict(), os.path.join(save_dir, "model1_last_epoch_weights.pth"))
    print("model2")
    val_loss = 0
    model2_train.eval()
    for iteration, (_, input2, targets) in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        with torch.no_grad():
            if cuda:
                input2  = input2.cuda()
                targets = [ann.cuda() for ann in targets]

            optimizer2.zero_grad()

            outputs         = model2_train(input2)

            loss_value_all  = 0

            for l in range(len(outputs)-3):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
            loss_value  = loss_value_all

        val_loss += loss_value.item()
    pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
    pbar.update(1)

    if 1:
        pbar.close()
        print('Finish Validation')
        loss_history2.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model2_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model2.state_dict(), os.path.join(save_dir, "model2_ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history2.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history2.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model2.state_dict(), os.path.join(save_dir, "model2_best_epoch_weights.pth"))
            
        torch.save(model2.state_dict(), os.path.join(save_dir, "model2_last_epoch_weights.pth"))