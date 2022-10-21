import torch, math
import torch.nn as nn
import numpy as np
from algorithm_models.make_VGG import make_vgg
# models for raw data



class raw_CNN(nn.Module):
    def __init__(self, class_num, time_stamp):
        nn.Module.__init__(self)

        # self.convs = my_resnet(layers=[2 ,2], layer_planes=[time_stamp, 128])
        self.convs = make_vgg(input_chnl=14, layers=[2, 3], layers_chnl=[time_stamp, 128])


        self.out = nn.Sequential(
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(256, class_num),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.convs(x)
        x = self.out(x)
        return x








































# models for feature_data
class my_CNN(nn.Module):
    def __init__(self):
        super(my_CNN,self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5) ,
            nn.Linear(128*10*4,128) ,
            nn.Dropout(p=0.51) ,
            nn.Linear(128,50)
        )

    def forward(self,x):
        x=x.unsqueeze(1)
        conv_out=self.conv(x)
        fc_out=self.fc(conv_out.view(x.shape[0], -1))
        return fc_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        # print(x.shape[0])
        y = self.avg_pool(x)

        # print("avg_pool",y.shape)
        y = self.fc(y.view(x.shape[0], -1))
        # print("fc",y.shape)
        # print("x",x.shape)
        return x * y.unsqueeze(-1)


class SELayer_datatype(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer_datatype, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, channel))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
            # nn.Dropout(0.5),
        )

    def forward(self, x):
        # print(x.shape[0])
        y = self.avg_pool(x)

        # print("avg_pool",y.shape)
        y = self.fc(y.view(x.shape[0], -1))
        # print("fc",y.shape)
        # print("x",x.shape)
        return x * y.unsqueeze(1)


class my_CNN_RNN_with_four_SENET(nn.Module):
    def __init__(self):
        super(my_CNN_RNN_with_four_SENET, self).__init__()
        self.lstm_att = SELayer(2)
        self.emg_att = SELayer_datatype(5, reduction=2) # 肌电
        self.gyr_att = SELayer_datatype(3, reduction=2) # 陀螺仪 自己的Transfomer自监督
        self.acc_att = SELayer_datatype(3, reduction=2) # 加速度
        self.lstm = nn.LSTM(128, 512, batch_first=True)
        self.emg_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )

        self.gyr_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )

        self.acc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(384 * 5 * 2, 512),
            nn.Dropout(p=0.51),
            nn.Linear(512, 128)
        )

        self.lstm_fc = nn.Sequential(

            nn.Linear(512, 300),

            nn.Linear(300, 289)
        )

    def forward(self, x):
        # print(x.shape)#x.shape:torch.Size([500, 48, 11])
        emg_x = x[:, :, 0:5]
        gyr_x = x[:, :, 5:8]
        acc_x = x[:, :, 8:11]
        emg_x = self.emg_att(emg_x)
        gyr_x = self.gyr_att(gyr_x)
        acc_x = self.acc_att(acc_x)
        x = torch.cat((emg_x, gyr_x, acc_x), axis=2)

        x = x.unsqueeze(1)
        # x.shape:torch.Size([500, 1, 48, 11])
        x1 = x[:, :, 0:30, :]
        x2 = x[:, :, 18:48, :]
        x = x1
        emg_conv_out = self.emg_conv(x[:, :, :, 0:5])
        gyr_conv_out = self.gyr_conv(x[:, :, :, 5:8])
        acc_conv_out = self.acc_conv(x[:, :, :, 8:11])
        # print(emg_conv_out.shape,acc_conv_out.shape,gyr_conv_out.shape)
        total_conv_out = np.hstack(
            (emg_conv_out.cpu().detach(), gyr_conv_out.cpu().detach(), acc_conv_out.cpu().detach()))
        total_conv_out = torch.from_numpy(total_conv_out).cuda()
        # print(total_conv_out.shape)
        fc_out1 = self.fc(total_conv_out.view(x.shape[0], -1))
        x = x2
        emg_conv_out = self.emg_conv(x[:, :, :, 0:5])
        gyr_conv_out = self.gyr_conv(x[:, :, :, 5:8])
        acc_conv_out = self.acc_conv(x[:, :, :, 8:11])
        # print(emg_conv_out.shape,acc_conv_out.shape,gyr_conv_out.shape)
        total_conv_out = np.hstack(
            (emg_conv_out.cpu().detach(), gyr_conv_out.cpu().detach(), acc_conv_out.cpu().detach()))
        total_conv_out = torch.from_numpy(total_conv_out).cuda()
        fc_out2 = self.fc(total_conv_out.view(x.shape[0], -1))
        att_input = torch.cat((fc_out1.unsqueeze(1), fc_out2.unsqueeze(1)), 1)
        lstm_input = self.lstm_att(att_input)
        lstm_output, (h0, c0) = self.lstm(lstm_input)
        # print(lstm_output.shape)
        # return self.lstm_fc(self.att[0]*lstm_output[:,0,:]+self.att[1]*lstm_output[:,1,:])
        return self.lstm_fc(lstm_output[:, -1, :])


class my_CNN_RNN(nn.Module):
    def __init__(self):
        super(my_CNN_RNN, self).__init__()
        # self.att=nn.Parameter(torch.Tensor([0.5,0.5]))
        self.lstm = nn.LSTM(128, 512, batch_first=True)
        self.emg_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )

        self.gyr_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )

        self.acc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(384 * 5 * 2, 512),
            nn.Dropout(p=0.51),
            nn.Linear(512, 128)
        )
        self.lstm_fc = nn.Sequential(

            nn.Linear(512, 256),

            nn.Linear(256, 311)
        )
        # self.softmax = nn.Sequential(nn.Dropout(0.5),nn.Linear(150,50))

    def forward(self, x):
        x = x.unsqueeze(1)
        # x.shape:torch.Size([500, 1, 48, 11])
        x1 = x[:, :, 0:30, :]
        x2 = x[:, :, 18:48, :]
        x = x1
        emg_conv_out = self.emg_conv(x[:, :, :, 0:5])
        gyr_conv_out = self.gyr_conv(x[:, :, :, 5:8])
        acc_conv_out = self.acc_conv(x[:, :, :, 8:11])
        # print(emg_conv_out.shape,acc_conv_out.shape,gyr_conv_out.shape)
        total_conv_out = np.hstack(
            (emg_conv_out.cpu().detach(), gyr_conv_out.cpu().detach(), acc_conv_out.cpu().detach()))
        total_conv_out = torch.from_numpy(total_conv_out).cuda()
        # print(total_conv_out.shape)
        fc_out1 = self.fc(total_conv_out.view(x.shape[0], -1))
        x = x2
        emg_conv_out = self.emg_conv(x[:, :, :, 0:5])
        gyr_conv_out = self.gyr_conv(x[:, :, :, 5:8])
        acc_conv_out = self.acc_conv(x[:, :, :, 8:11])
        # print(emg_conv_out.shape,acc_conv_out.shape,gyr_conv_out.shape)
        total_conv_out = np.hstack(
            (emg_conv_out.cpu().detach(), gyr_conv_out.cpu().detach(), acc_conv_out.cpu().detach()))
        total_conv_out = torch.from_numpy(total_conv_out).cuda()
        fc_out2 = self.fc(total_conv_out.view(x.shape[0], -1))
        lstm_input = torch.cat((fc_out1.unsqueeze(1), fc_out2.unsqueeze(1)), 1)
        lstm_output, (h0, c0) = self.lstm(lstm_input)
        # print(lstm_output.shape)
        # return self.lstm_fc(self.att[0]*lstm_output[:,0,:]+self.att[1]*lstm_output[:,1,:])
        return self.lstm_fc(lstm_output[:, -1, :])

class my_CNN_RNN_withSENET_datatypeSENET(nn.Module):
    def __init__(self):
        super(my_CNN_RNN_withSENET_datatypeSENET, self).__init__()
        self.lstm_att = SELayer(2)
        self.datatype_att = SELayer_datatype(11, reduction=4)
        self.lstm = nn.LSTM(128, 512, batch_first=True)
        self.emg_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )

        self.gyr_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )

        self.acc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(384 * 5 * 2, 512),
            nn.Dropout(p=0.51),
            nn.Linear(512, 128)
        )
        #         self.lstm_fc=nn.Sequential(

        #             nn.Linear(512,128) ,

        #             nn.Linear(128,50)
        #         )
        self.lstm_fc = nn.Sequential(

            nn.Linear(512, 300),

            nn.Linear(300, 289)
        )
        # self.softmax = nn.Sequential(nn.Dropout(0.5),nn.Linear(150,50))

    def forward(self, x):
        # print(x.shape)#x.shape:torch.Size([500, 48, 11])
        x = self.datatype_att(x)
        x = x.unsqueeze(1)
        # x.shape:torch.Size([500, 1, 48, 11])
        x1 = x[:, :, 0:30, :]
        x2 = x[:, :, 18:48, :]
        x = x1
        emg_conv_out = self.emg_conv(x[:, :, :, 0:5])
        gyr_conv_out = self.gyr_conv(x[:, :, :, 5:8])
        acc_conv_out = self.acc_conv(x[:, :, :, 8:11])
        # print(emg_conv_out.shape,acc_conv_out.shape,gyr_conv_out.shape)
        total_conv_out = np.hstack(
            (emg_conv_out.cpu().detach(), gyr_conv_out.cpu().detach(), acc_conv_out.cpu().detach()))
        total_conv_out = torch.from_numpy(total_conv_out).cuda()
        # print(total_conv_out.shape)
        fc_out1 = self.fc(total_conv_out.view(x.shape[0], -1))
        x = x2
        emg_conv_out = self.emg_conv(x[:, :, :, 0:5])
        gyr_conv_out = self.gyr_conv(x[:, :, :, 5:8])
        acc_conv_out = self.acc_conv(x[:, :, :, 8:11])
        # print(emg_conv_out.shape,acc_conv_out.shape,gyr_conv_out.shape)
        total_conv_out = np.hstack(
            (emg_conv_out.cpu().detach(), gyr_conv_out.cpu().detach(), acc_conv_out.cpu().detach()))
        total_conv_out = torch.from_numpy(total_conv_out).cuda()
        fc_out2 = self.fc(total_conv_out.view(x.shape[0], -1))
        att_input = torch.cat((fc_out1.unsqueeze(1), fc_out2.unsqueeze(1)), 1)
        lstm_input = self.lstm_att(att_input)
        lstm_output, (h0, c0) = self.lstm(lstm_input)
        # print(lstm_output.shape)
        # return self.lstm_fc(self.att[0]*lstm_output[:,0,:]+self.att[1]*lstm_output[:,1,:])
        return self.lstm_fc(lstm_output[:, -1, :])

# models for feature_data