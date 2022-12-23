from utils_dyn import *
from model_dyn import *
import argparse
import warnings
warnings.filterwarnings("ignore")

set_deterministic()

DATA_FILE = "/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/data/RESIN_MH_001.csv"


parser = argparse.ArgumentParser() 
parser.add_argument('--model_name', type=str, default = 'mixed_teacher_forcing')
parser.add_argument('--train_fraction', type=float, default = 0.8)
parser.add_argument('--inp_wnd', type=int, default = 7)
parser.add_argument('--out_wnd', type=int, default = 7)
parser.add_argument('--stride', type=int, default = 1)
parser.add_argument('--lstm_hid_dim', type=int, default = 50)
args=parser.parse_args()

train_df, val_df, test_df = train_val_test_split(DATA_FILE,args.train_fraction)

MODELS_PATH = "snapshots/" + str(args.model_name)+ str(args.train_fraction) + "_" + str(args.inp_wnd) + "_" + str(args.out_wnd) +"_" + str(args.stride) + "_" + str(args.lstm_hid_dim) +  "/"
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

f = open(MODELS_PATH + "logs.txt", "w")

traindataset = CustomDataset(train_df,args.inp_wnd,args.out_wnd,args.stride)
valdataset = CustomDataset(val_df,args.inp_wnd,args.out_wnd,args.stride)
testdataset = CustomDataset(test_df,args.inp_wnd,args.out_wnd,args.stride)

#args.inp_wnd = 8
BATCH_SIZE = 32
N_EPOCHS = 121
TEACHER_FORCING_RATIO = 0.25
LR = 0.01
DYNAMIC_TF = True
TARGET_LEN = args.out_wnd
MIN_INDEX_LOSS = 0

trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(valdataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)


model = lstm_seq2seq(traindataset.num_features, args.lstm_hid_dim)
optimizer = optim.Adam(model.parameters(), lr = LR)
criterion = nn.MSELoss()

print("Training the model")

model = model.float()

train_loss = []
val_loss = []
test_loss = []

for epoch in tqdm(range(N_EPOCHS)) : 
    model.train()
    avg_batch_loss = 0.0
    n_batches = 0

    for (input_batch,target_batch) in trainloader : 

        input_batch = input_batch.view(input_batch.shape[1],input_batch.shape[0],input_batch.shape[2])
        target_batch = target_batch.view(target_batch.shape[1],target_batch.shape[0],target_batch.shape[2])
        # outputs tensor
        
        outputs = torch.zeros(TARGET_LEN, input_batch.shape[1], input_batch.shape[2])

        # initialize hidden state
        encoder_hidden = model.encoder.init_hidden(input_batch.shape[1])

        # zero the gradient
        optimizer.zero_grad()

        # encoder outputs
        encoder_output, encoder_hidden = model.encoder(input_batch.float())

        # decoder with teacher forcing
        decoder_input = input_batch[-1, :, :].float()   # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden

        if args.model_name == 'recursive':
            # predict recursively
            for t in range(TARGET_LEN): 
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output

        if args.model_name  == 'teacher_forcing':
            # use teacher forcing
            if random.random() < TEACHER_FORCING_RATIO:
                for t in range(TARGET_LEN): 
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    decoder_input = target_batch[t, :, :].float()

            # predict recursively 
            else:
                for t in range(TARGET_LEN): 
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    decoder_input = decoder_output

        if args.model_name  == 'mixed_teacher_forcing':
            # predict using mixed teacher forcing
            for t in range(TARGET_LEN):
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                
                # predict with teacher forcing
                if random.random() < TEACHER_FORCING_RATIO:
                    decoder_input = target_batch[t, :, :].float()
                
                # predict recursively 
                else:
                    decoder_input = decoder_output

        # compute the loss 
        loss = criterion(outputs, target_batch.float())
        avg_batch_loss += loss.item()
        
        # backpropagation
        loss.backward()
        optimizer.step()

        n_batches+=1

    # loss for epoch 
    avg_batch_loss /= n_batches 

    # dynamic teacher forcing
    if DYNAMIC_TF and TEACHER_FORCING_RATIO > 0:
        TEACHER_FORCING_RATIO = TEACHER_FORCING_RATIO - 0.02 

    model.eval()
    
    avg_val_batch_loss = evaluate(model,valloader,criterion,TARGET_LEN)
    avg_test_batch_loss = evaluate(model,testloader,criterion,TARGET_LEN)

    f.write("Epoch : {} Train Avg Loss: {:.4} \n".format(epoch, avg_batch_loss))
    f.write("Epoch : {} Val Avg Loss: {:.4} \n \n".format(epoch, avg_val_batch_loss))

    if(epoch>MIN_INDEX_LOSS) : 
        train_loss.append(avg_batch_loss/5.0)
        val_loss.append(avg_val_batch_loss)
        test_loss.append(avg_test_batch_loss)

    if(epoch%10 == 0) : 
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(np.arange(MIN_INDEX_LOSS,MIN_INDEX_LOSS+len(train_loss)),train_loss,label='Training Loss', color = 'red')
        plt.plot(np.arange(MIN_INDEX_LOSS,MIN_INDEX_LOSS+len(val_loss)),val_loss,label='ValLoss', color = 'blue')
        plt.plot(np.arange(MIN_INDEX_LOSS,MIN_INDEX_LOSS+len(test_loss)),test_loss,label='TestLoss', color = 'green')
        plt.savefig(MODELS_PATH + 'loss.png')
        torch.save(model,MODELS_PATH  + "epoch_" + str(epoch))
        print("Epoch : {} Val Avg Loss: {:.4} \n \n".format(epoch, avg_val_batch_loss))




