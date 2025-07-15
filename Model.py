import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.shape  #[batch,n_features,a,b]
    y_shapes = y.shape  #[batch,n]
    y2 = y.view(x_shapes[0],y_shapes[1],1,1)                              #[batch,n,1,1]
    y2 = y2.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])      #[batch,n,a,b]

    return torch.cat((x, y2),dim=1)                                     #[batch,n_features+n,a,b]

def conv_prev_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.shape  #[batch,n_features,a,b]
    y_shapes = y.shape  #[batch,16,a,b]
    if x_shapes[2:] == y_shapes[2:]:
        y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])  #[batch,16,a,b]

        return torch.cat((x, y2),dim=1)                                 #[batch,n_features+16,a,b]

    else:
        print(x_shapes[2:])
        print(y_shapes[2:])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)

#Temporary
class Generator(nn.Module):

    def __init__(self, input_size, cond_1d_size, instrument_size=1, n_hlayers=128):
            super().__init__()

            self.input_size = input_size
            self.instrument_size = instrument_size
            self.cond1d_dim = cond_1d_size

            #generator layers
            #as said in the DCGAN paper always ReLU activation function in the generator excluded the last layer
            #as said in the DCGAN paper always batchnorm iin the generator excluded the last layer
            self.ff1 = nn.Sequential(
                nn.Linear(input_size+cond_1d_size, 1024),                                                               #[batch,1024]
                nn.BatchNorm1d(1024),
                nn.ReLU()
                )
            self.ff2 = nn.Sequential(
                nn.Linear(1024+cond_1d_size,n_hlayers*2),                                                                                    #[batch,512]
                nn.BatchNorm1d(n_hlayers*2),
                nn.ReLU()
                )
            #reshape to [batch size,128,1,2]
            # #+condition [batch,128+cond_1d_size+16,1,2]
            self.cnn1 = nn.Sequential(
                nn.ConvTranspose2d(n_hlayers+cond_1d_size+16, n_hlayers, kernel_size=(1,2), stride=(2,2), bias=False, padding=0),           #[batch,128,1,4]
                nn.BatchNorm2d(n_hlayers),
                nn.ReLU()
                )
            #+condition [batch,128+cond_1d_size+16,1,2]
            self.cnn2 = nn.Sequential(
                nn.ConvTranspose2d(n_hlayers+cond_1d_size+16, n_hlayers, kernel_size=(1,2), stride=(2,2), bias=False, padding=0),           #[batch,128,1,8]
                nn.BatchNorm2d(n_hlayers),
                nn.ReLU()
                )
            #+condition [batch,128+cond_1d_size+16,1,2]
            self.cnn3 = nn.Sequential(
                nn.ConvTranspose2d(n_hlayers+cond_1d_size+16, n_hlayers, kernel_size=(1,2), stride=(2,2), bias=False, padding=0),           #[batch,128,1,16]
                nn.BatchNorm2d(n_hlayers),
                nn.ReLU()
                )
            #+condition [batch,128+cond_1d_size+16,1,2]
            self.cnn4 = nn.Sequential(
                nn.ConvTranspose2d(n_hlayers+cond_1d_size+16, instrument_size, kernel_size=(128,1), stride=(2,1), bias=False, padding=0),       #[batch,instrument_size,128,16]
                nn.Sigmoid()
                #Sigmoid funciotn because we want to generate the matrixes of music without velocity, i.e. only (0,1)
                #Thus we use the sigmoid which is a smoother version of the sign function
                )
            #conditioner layers
            # #as in Midinet model we use the Leaky activation funciton for the conditioner
            self.h0_prev = nn.Sequential(
                nn.Conv2d(in_channels=instrument_size, out_channels=16, kernel_size=(128,1), stride=(2,1)),                  #[batch,16,1,16]
                nn.BatchNorm2d(16),
                nn.LeakyReLU()          #note: in the original paper leak=0.2, default leak=0.01
                )
            self.h1_prev = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,2), stride=(2,2)),                                  #[batch,16,1,8]
                nn.BatchNorm2d(16),
                nn.LeakyReLU()
                )
            self.h2_prev = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,2), stride=(2,2)),                                  #[batch,16,1,4]
                nn.BatchNorm2d(16),
                nn.LeakyReLU()
                )
            self.h3_prev = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,2), stride=(2,2)),                                  #[batch,16,1,2]
                nn.BatchNorm2d(16),
                nn.LeakyReLU()
                )

    def forward(self, z, prev_bar, cond_1d, batch_size):

            #2d condiiton
            cond0 = self.h0_prev(prev_bar)          #[batch,16,1,16]
            cond1 = self.h1_prev(cond0)             #[batch,16,1,8]
            cond2 = self.h2_prev(cond1)             #[batch,16,1,4]
            cond3 = self.h3_prev(cond2)             #[batch,16,1,2]

            #single cond_1d size =[n,1], batch_cond_1d size = [batch_size,n]

            input = torch.cat((z,cond_1d), dim=1)   #[batch_size, input_size+cond_1d_size]

            h0 = self.ff1(input)                    #[batch,1024]
            h0 = torch.cat((h0,cond_1d), dim=1)     #[batch,1024+cond_1d_size]

            h1 = self.ff2(h0)                       #[batch,256]
            h1 = h1.reshape(batch_size, 256, 1, 2)  #[batch,128,1,2]
            h1 = conv_cond_concat(h1,cond_1d)       #[batch,128+cond_1d_size,1,2]
            h1 = conv_prev_concat(h1,cond3)         #[batch,128+cond_1d_size+16,1,2]

            h2 = self.cnn1(h1)                      #[batch,128,1,4]
            h2 = conv_cond_concat(h2,cond_1d)       #[batch,128+cond_1d_size,1,4]
            h2 = conv_prev_concat(h2,cond2)         #[batch,128+cond_1d_size+16,1,4]

            h3 = self.cnn2(h2)                      #[batch,128,1,8]
            h3 = conv_cond_concat(h3,cond_1d)       #[batch,128+cond_1d_size,1,8]
            h3 = conv_prev_concat(h3,cond1)         #[batch,128+cond_1d_size+16,1,8]

            h4 = self.cnn3(h3)                      #[batch,128,1,16]
            h4 = conv_cond_concat(h4,cond_1d)       #[batch,128+cond_1d_size,1,16]
            h4 = conv_prev_concat(h4,cond0)         #[batch,128+cond_1d_size+16,1,16]

            out = self.cnn4(h4)                     #[batch,instrument_size,128,16]

            return out
    

    
    


class Discriminator(nn.Module):

    def __init__(self, cond_1d_size, instrument_size=1):
        super().__init__()

        self.instrument_size = instrument_size
        self.cond1d_dim = cond_1d_size

        #as said in the DCGAN paper always batchnorm in the discriminator layers excluded the first layer
        self.cnn1 = nn.Sequential(
            nn.Conv2d(2*instrument_size+cond_1d_size, 32, kernel_size=(128,2), stride=(2,2), padding=0),        #[batch,32,1,8]
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        #+condition [batch,14+cond_1d_size,1,8]
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32+cond_1d_size, 77, kernel_size=(1,4), stride=2, padding=0),                             #[batch,77,1,3]
            #Adding residual block
            nn.LeakyReLU(),
            nn.Conv2d(77, 77, kernel_size=(1,1)),  # Identity shortcut
            nn.LeakyReLU()
        )

        self.ffnn1 = nn.Sequential(
             #+condition [batch,231+cond_1d_size]
            nn.Linear(231+cond_1d_size, 1024),
            #nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        #+condition [batch,1024+cond_1d_size]
        self.ffnn2 = nn.Linear(1024+cond_1d_size, 1)      #no sigmoid activation function because it is already in the definition of the cross entropy loss function


    def forward(self, x, prev_bar, cond_1d):
        input = conv_cond_concat(x,cond_1d)         #[batch,instrument_size+cond_1d_size,128,16]
        input = conv_prev_concat(input,prev_bar)    #[batch,2*instrument_size+cond_1d_size,128,16]

        h0 = self.cnn1(input)                       #[batch,14,1,8]
        fm = h0
        h0 = conv_cond_concat(h0, cond_1d)          #[batch,14+cond_1d_size,1,8]

        h1 = self.cnn2(h0)                          #[batch,77,1,3]
        h1 = torch.flatten(h1, 1)                   #[batch,77*3*1]
        h1 = torch.cat((h1,cond_1d),dim=1)          #[batch,231+cond_1d_size]

        h2 = self.ffnn1(h1)                         #[batch,1024]
        h2 = torch.cat((h2,cond_1d),dim=1)          #[batch,1024+cond_1d_size]

        h3 = self.ffnn2(h2)                         #[batch,1]
        h3_sigmoid = torch.sigmoid(h3)


        return h3_sigmoid, h3, fm








#Model for genre recognition
class GenreCNN(nn.Module):
                        #Classify in 10 classes the genre 
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  
            nn.Flatten(),
            nn.Linear(4*4*64, 128),     
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)  
        return x