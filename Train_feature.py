import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader as Dataloader
from optimizers import get_optimizer, LR_Scheduler


from loss import negative_cosine_similarity
from Model_feature import SimSiam
from DataLoader_feature import SpecData


if __name__ == '__main__':
    model = SimSiam(
        latent_dim=256,
        proj_hidden_dim=256,
        pred_hidden_dim=128
    )

    train_data = SpecData("train")
    # test_data = SpecData("test")

    train_dataloader = Dataloader(train_data,
                                    batch_size = 32,
                                    shuffle = True,
                                    num_workers = 0)

    # test_dataloader = Dataloader(test_data,
    #                    batch_size = 2,
    #                    shuffle =False,
    #                    num_workers = 1)
    savePath = '/home/lijingru/bishe/Experiment/runs/Apr28_19-47-19_localhost.localdomain/Simsiam_400.pt'
    model.encoder.load_state_dict(torch.load(savePath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    print(device)
    model = model.to(device)
    model.train()

    name = 'sgd'
    momentum = 0.9
    weight_decay = 0.0005

    warmup_epochs = 0
    warmup_lr = 0
    batch_size = 32
    num_epochs = 800
    base_lr = 3
    final_lr = 0

    # need import
    optimizer = get_optimizer(
        name, model, 
        lr=base_lr*batch_size/256, 
        momentum=momentum,
        weight_decay=weight_decay)


    # need import
    lr_scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs, warmup_lr*batch_size/256, 
        num_epochs, base_lr*batch_size/256, final_lr*batch_size/256, 
        len(train_dataloader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    writer = SummaryWriter()
    
    n_iter = 0

    for epoch in range(800):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=False)

        for ii,(spec, resamp_spec) in pbar:
            # encode
            e1, e2 = model.encode(spec.cuda()), model.encode(resamp_spec.cuda())

            # project
            z1, z2 = model.project(e1), model.project(e2)

            # predict
            p1, p2 = model.predict(z1), model.predict(z2)

            # compute loss
            loss1 = negative_cosine_similarity(p1, z2)
            loss2 = negative_cosine_similarity(p2, z1)
            loss = loss1/2 + loss2/2
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_description("Epoch {}, Loss: {:.4f}".format(epoch, float(loss)))

            log_interval = 100
            if n_iter % log_interval == 0:
                writer.add_scalar(tag="loss", scalar_value=float(loss), global_step=n_iter)

            n_iter += 1

        # print(writer.log_dir)
        torch.save(model.encoder.state_dict(), os.path.join(writer.log_dir,  'Simsiam_' + str(epoch) + ".pt"))
