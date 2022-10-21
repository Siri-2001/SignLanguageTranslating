import os
from common.metrics import *
from common.Logging import Log
import datetime, time
def evaluate(model, test_iter, loss_function, device):
    eval_total_loss = 0
    eval_total_sample = 0
    eval_total_correctnum = 0
    with torch.no_grad():
        model.eval()
        for x, y in test_iter:
            y_h = model(x.to(device))
            loss = loss_function(y_h, y.to(device))
            eval_total_correctnum += count_correct(y_h, y)
            eval_total_loss += loss.item()
            eval_total_sample += x.shape[0]
            del x, y_h
            torch.cuda.empty_cache()
    return eval_total_loss/eval_total_sample,eval_total_correctnum/eval_total_sample


def Train(args, model, epoch_num, train_iter, test_iter, loss_function, optimizer):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    ckpts_file = str(datetime.datetime.now()).replace(' ', '').replace('-', '').replace(':', '_').replace('.', '_')
    if os.path.exists(os.path.join(args.checkpoint_path, ckpts_file)):
        exit('Too Fast!!!')
    os.mkdir(os.path.join(args.checkpoint_path, ckpts_file))
    print(model, file=open(os.path.join(args.checkpoint_path,ckpts_file, 'model_config.txt'),'w'))
    for i in range(1, epoch_num+1):
        train_total_loss = 0
        train_sample_num = 0
        train_total_correctnum = 0
        epoch_start_time = time.time()
        for x, y in train_iter:
            model.train()
            optimizer.zero_grad()
            y_h = model(x.to(device))
            loss = loss_function(y_h, y.to(device))
            loss.backward()
            train_total_correctnum += count_correct(y_h, y)
            train_total_loss += loss.item()
            train_sample_num += x.shape[0]
            optimizer.step()
            del x, y_h
            torch.cuda.empty_cache()
        if i % args.sckp_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, ckpts_file,
                                                        str(datetime.datetime.now()).replace(' ', '')
                                                        .replace('-', '').replace(':', '_').replace('.', '_')))
        avg_train_loss = train_total_loss/train_sample_num
        train_acc = train_total_correctnum/train_sample_num
        avg_eval_loss, eval_acc = evaluate(model, test_iter, loss_function, device)
        epoch_end_time = time.time()
        Log('epoch %s: (time: %s) Train Loss:%s Eval Loss:%s '
            'Train Acc:%.2f Eval Acc:%.2f ' % (str(i),
                                               str(epoch_end_time-epoch_start_time),
                                               avg_train_loss, avg_eval_loss,
                                               train_acc, eval_acc),
                                               os.path.join(args.checkpoint_path,
                                                            ckpts_file, 'log'+str(datetime.datetime.now()).replace(' ', '')
                                                        .replace('-', '').replace(':', '_').replace('.', '_'))).log()
    return







