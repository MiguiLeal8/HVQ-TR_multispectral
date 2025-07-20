import argparse
import logging
import os
import pprint
import shutil
import time
import pandas as pd

import torch
import torch.distributed as dist
import torch.optim
import yaml
from ..datasets.data_builder import build_dataloader
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from ..utils.criterion_helper import build_criterion
from ..utils.dist_helper import setup_distributed
from ..utils.eval_helper import dump, log_metrics, merge_together, performances
from ..utils.lr_helper import get_scheduler
from ..utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
)
from ..utils.optimizer_helper import get_optimizer
from ..utils.vis_helper import visualize_compound, visualize_single
from ..models.HVQ_TR_switch import HVQ_TR_switch
# from models.HVQ_TR_switch_OT import HVQ_TR_switch_OT

from ..datasets.datasets_muestras import *
import math,random,struct,os,time,sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn import preprocessing
import torchvision.transforms as transforms

from skimage.filters import threshold_otsu, threshold_li, threshold_isodata
from kneed import KneeLocator
import matplotlib.pyplot as plt

# z1
#DATASET='../../../Img/z1AD/z1.raw'
#GT='../../../Img/z1AD/z1_gt_reetiquetado.pgm'

# z2
#DATASET='../../../Img/z2AD/z2.raw'
#GT='../../../Img/z2AD/z2_gt_reetiquetado.pgm'

# e1
#DATASET='../../../Img/e1/data/e1/e1.raw'
#GT='../../../Img/e1/data/e1/e1_gt.pgm'

# e2
DATASET='../../../Img/e2/data/e2/e2.raw'
GT='../../../Img/e2/data/e2/e2_gt.pgm'

SAMPLES=[0.01,0.01] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.02,0.01) 
PAD=1  # hacemos padding en los bordes para aprovechar todas las muestras
DET=0  # experimentos: 0-aleatorios, 1-deterministas

parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument('--train_only_four_decoder',default=False,type=bool)


def main():
    np.set_printoptions(threshold=np.inf)

    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    time_start=time.time()
    time_start_train_val=time.time()
    # Device configuration
    cuda=True if torch.cuda.is_available() else False
    device=torch.device('cuda' if cuda else 'cpu')
    print("DEVICE: ", device)
    if torch.backends.cudnn.is_available():
        print('* Activando CUDNN')
        torch.backends.cudnn.enabled=True
        torch.backends.cudnn.beBhmark=True
    # experimentos deterministas o aleatorios
    if(DET==1):
        SEED=0
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        if(cuda==False):
            torch.use_deterministic_algorithms(True)
            g=torch.Generator(); g.manual_seed(SEED)
        else:
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark=False

    #COMPROBACIONES DE PROCESOS
    num_gpus = torch.cuda.device_count()
    print(f"Número de GPUs disponibles: {num_gpus}")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()  # Número total de procesos
        rank = dist.get_rank()  # Rango del proceso actual
        print(f"Proceso {rank+1} de {world_size}")

    # Load datos
    (datos,H,V,B)=read_raw(DATASET)     # datos[V][H][B]
    (truth,H1,V1)=read_pgm(GT)          # truth[]       (valores: 1 y 2)
    # durante la ejecucion de la red vamos a coger patches de tamano cuadrado
    sizex=32; sizey=32
    print("TRUTH antes del padding: ", len(truth))

    # En z1 hay pixeles de recortes que no corresponden a datos normales ni anomalos y pueden influir negativamente en el entrenamiento
    print("Eliminando pixeles de recortes...")
    for i in range(datos.size(0)):
        for j in range(datos.size(1)):
            if datos[i][j][0] == 0 and datos[i][j][1] == 0 and datos[i][j][2] == 0 and datos[i][j][3] == 0 and datos[i][j][4] == 0:
                truth[i*datos.size(1)+j]=0

    # hacemos padding en el dataset para poder aprovechar hasta el borde
    if(PAD):
        datos=torch.FloatTensor(np.pad(datos,((sizey//2,sizey//2),(sizex//2,sizex//2),(0,0)),'symmetric'))
        H=H+2*(sizex//2); V=V+2*(sizey//2)
        truth=np.reshape(truth,(-1,H1))
        truth=np.pad(truth,((sizey//2,sizey//2),(sizex//2,sizex//2)),'constant')
        H1=H1+2*(sizex//2); V1=V1+2*(sizey//2)
        truth=np.reshape(truth,(H1*V1))
    # necesitamos los datos en band-vector (B, V, H)
    datos=np.transpose(datos,(2,0,1))

    # Guardar los valores de truth en un archivo .txt
    '''truth_array = np.array(truth)  # Asegúrate de que 'truth' sea un arreglo de NumPy
    truth_minus_one = np.maximum(truth_array -1, 0)
    np.savetxt("truth_z.txt", truth_minus_one, fmt="%d")'''

    print("TRUTH despues del padding: ", len(truth))

    # Selection training, testing sets
    (train,val,test,nclases,nclases_no_vacias)=select_training_samples(truth,H,V,sizex,sizey,SAMPLES)
    
    dataset_train=HyperDataset(datos,truth,train,H,V,sizex,sizey)
    print('  - train dataset:',len(dataset_train))
    dataset_test=HyperDataset(datos,truth,test,H,V,sizex,sizey)
    print('  - test dataset:',len(dataset_test))

    # Dataloader
    batch_size=100 # defecto 100
    train_loader=DataLoader(dataset_train,batch_size,num_workers=4,pin_memory=True, sampler=DistributedSampler(dataset_train, shuffle=True))
    test_loader=DataLoader(dataset_test,batch_size,num_workers=4,pin_memory=True, sampler=DistributedSampler(dataset_test, shuffle=False))

    # Si queremos validacion
    if(len(val)>0):
        dataset_val=HyperDataset(datos,truth,val,H,V,sizex,sizey)
        print('  - val dataset:',len(dataset_val))
        val_loader=DataLoader(dataset_val,batch_size,num_workers=4,pin_memory=True,sampler=DistributedSampler(dataset_val, shuffle=False))
    

    # create model
    model = HVQ_TR_switch(channel=272, embed_dim=256)
    # C
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.to(device)
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        # Without GPU: use the model in CPU without DDP
        model.to(device)

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    parameters = [
        {'params': filter(lambda p: p.requires_grad, model.parameters())}
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)

    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    resume_model = config.saver.get("resume_model", None)

    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)

    if args.evaluate:
        validate(val_loader, model, batch_size)
        return

    criterion = build_criterion(config.criterion)
    
    # results = 0 -> Entrenar el modelo y calcular las predicciones
    # results = 1 -> Obtener las predicciones de un archivo .txt
    results = 1

    if results == 0:
        # checkpoint = 0 -> Entrenar el modelo
        # checkpoint = 1 -> Obtener un modelo ya entrenado y calcular las predicciones
        checkpoint = 0
        if checkpoint == 0:
        
            for epoch in range(last_epoch, config.trainer.max_epoch):
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
                last_iter = epoch * len(train_loader)
                train_one_epoch(
                    train_loader,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    last_iter,
                    tb_logger,
                    criterion,
                    frozen_layers,
                    batch_size,
                )

                lr_scheduler.step()

                if (epoch) % config.trainer.val_freq_epoch == 0:
                    ret_metrics, threshold = validate(val_loader, model, batch_size)

                    if rank == 0:
                        ret_key_metric = ret_metrics[key_metric]
                        print('Epoch :',epoch + 1,'Best Metric:',best_metric,'Current Metric:',ret_key_metric)
                        is_best = ret_key_metric >= best_metric
                        best_metric = max(ret_key_metric, best_metric)
                        save_checkpoint(
                            {
                                "epoch": epoch + 1,
                                "arch": config.net,
                                "state_dict": model.state_dict(),
                                "best_metric": best_metric,
                                "optimizer": optimizer.state_dict(),
                            },
                            is_best,
                            config,
                        )
                    
        
        if checkpoint == 1:
            checkpoint_path="../../../Mod/model_hvq.ckpt"
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        
        time_end_train_val=time.time()
        print('* Training and validation time: %.0f s'%(time_end_train_val - time_start_train_val))


        # Test the model
        print("THRESHOLD: ", threshold)
        output=np.zeros(H*V,dtype=np.uint8)
        label_array=np.zeros(H*V,dtype=np.uint8)# etiquetas
        preds=np.zeros(H*V,dtype=np.uint8) # predicciones finales (0=normal, 1=anomalia)
        pred_mean=np.zeros(H*V,dtype=np.float32) # predicciones antes de comparar con el umbral para guardar en .txt
        preds_list = [] # almacenar el resultado en un .txt
        
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        model.eval()
        with torch.no_grad():
            correct=0; total=0; recoge=[]

            index=[]
            acc=0 
            for i, (inputs,labels) in enumerate(test_loader):
                inputs=inputs.to(device)
                labels=labels.to(device)
                
                for k in range(labels.size(0)):
                    label_array[test[k+acc]]=int(labels[k])
                acc+=labels.size(0)

                height = torch.full((batch_size,), sizey, device=device)
                width = torch.full((batch_size,), sizex, device=device)

                mask = torch.zeros(labels.size(0), 1, sizey, sizex)
                for k in range(labels.size(0)):
                    if labels[k] == 0:
                        mask[k] = torch.zeros((1, height[k], width[k]))
                    else:
                        mask[k] = torch.ones((1, height[k], width[k]))

                filenames = [f"patch/{i}_{j}.png" for j in range(labels.size(0))]
                
                input= {
                    "filename": filenames,
                    "image": inputs,
                    "mask": mask,
                    "height": height,
                    "width": width,
                    "label": labels,
                    "clsname": ["hyperspectral"] * labels.size(0),
                }

                outputs=model(input)
                
                pred=outputs['pred']
                            
                # Hace la media de los pixeles
                pred = pred.reshape(pred.size(0), -1).mean(axis=1) 
                # Selecciona el pixel central
                #pred=pred[:,0,16,16] 

                predicted = (pred > threshold).float() # se compara con el umbral para obtener valores binarios
                
                correct+=(predicted==labels).sum().item()

                predicted_cpu=predicted.cpu() 
                pred_cpu=pred.cpu() 
                for k in range(len(predicted_cpu)):
                    output[test[total+k]]=np.uint8(predicted_cpu[k]+1)
                    preds[test[total+k]]=np.uint8(predicted_cpu[k])
                    pred_mean[test[total+k]]=np.float32(pred_cpu[k])
                    index.append(test[total+k])
                total+=labels.size(0)
                if(total%10000==0): print('  Test:',total,'/',len(dataset_test))

        np.savetxt("pred_mean_test.txt", pred_mean, fmt="%f")

        if(PAD):
            label_array=np.reshape(label_array,(-1,H1))
            label_array=label_array[sizey//2:V1-sizey//2,sizex//2:H1-sizex//2]
            preds=np.reshape(preds,(-1,H1))
            preds=preds[sizey//2:V1-sizey//2,sizex//2:H1-sizex//2]
            pred_mean=np.reshape(pred_mean,(-1,H1))
            pred_mean=pred_mean[sizey//2:V1-sizey//2,sizex//2:H1-sizex//2]

            H1=H1-2*(sizex//2); V1=V1-2*(sizey//2)

            label_array=np.reshape(label_array,(H1*V1))
            preds=np.reshape(preds,(H1*V1))
            pred_mean=np.reshape(pred_mean,(H1*V1))

    if results == 1:
        
        # Si se obtienen los datos de ficheros externos, no son tensores de PyTorch, asi que no es necesario pasarlos a la CPU
        label_array = np.loadtxt("truth_e2.txt", dtype = int).astype(np.uint8)
        pred_mean = np.loadtxt("pred_mean_test.txt", dtype = float).astype(np.float32)
        
        #THRESHOLD:
        #OTSU
        #threshold = threshold_otsu(pred_mean)
        #LI
        #threshold = threshold_li(pred_mean)
        #ISODATA
        #threshold = threshold_isodata(pred_mean)
        #KNEE
        #sorted_data = np.sort(pred_mean); x = np.arange(len(sorted_data)); knee_locator = KneeLocator(x, sorted_data, curve='convex', direction='increasing'); threshold = sorted_data[knee_locator.knee] if knee_locator.knee is not None else threshold_otsu(pred_mean)
        #YOUDEN'S J STATIC
        fpr, tpr, thresholds = metrics.roc_curve(label_array, pred_mean, pos_label=1); j_scores = tpr - fpr; optimal_idx = np.argmax(j_scores); threshold = thresholds[optimal_idx]

        '''
        # GRAFICA DE VALORES DE PIXEL ORDENADOS, CON EL THRESHOLD MARCADO
        # Ordenar los scores para visualizarlos
        sorted_data = np.sort(pred_mean)
        x = np.arange(len(sorted_data))
        # Obtener la posición aproximada del threshold en los datos ordenados
        threshold_idx = np.searchsorted(sorted_data, threshold)
        # Plot ordenado con threshold marcado
        plt.figure(figsize=(8, 4))
        plt.plot(x, sorted_data, label='Scores ordenados')
        plt.axvline(threshold_idx, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
        plt.title("Curva ordenada de valores de predicción")
        plt.xlabel("Índice")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        '''

        print("THRESHOLD: ", threshold)
        
        preds = (pred_mean > threshold).astype(np.uint8)

        output = np.zeros_like(label_array, dtype=np.uint8)
        output = preds + 1

        if(PAD):
            label_array=np.reshape(label_array,(-1,H1))
            label_array=label_array[sizey//2:V1-sizey//2,sizex//2:H1-sizex//2]
            preds=np.reshape(preds,(-1,H1))
            preds=preds[sizey//2:V1-sizey//2,sizex//2:H1-sizex//2]
            pred_mean=np.reshape(pred_mean,(-1,H1))
            pred_mean=pred_mean[sizey//2:V1-sizey//2,sizex//2:H1-sizex//2]

            H1=H1-2*(sizex//2); V1=V1-2*(sizey//2)

            label_array=np.reshape(label_array,(H1*V1))
            preds=np.reshape(preds,(H1*V1))
            pred_mean=np.reshape(pred_mean,(H1*V1))

        pred_mean=pred_mean.reshape(V1, H1)
        #show_hist(pred_mean) # mostrar el histograma con las predicciones
        #save_grey_scale_image(pred_mean) # guardar el resultado de las predicciones sobre una imagen en escala de grises

        correct = np.sum(preds == label_array)
        total = len(label_array)
        

    print('* Accuracy: %02.02f'%(100*correct/total))

    save_anomaly_detection_as_rgb_image(preds, label_array, V1, H1) 

    # precisiones a nivel de clase
    correct=0; total=0; AA=0; OA=0
    class_correct=[0]*(nclases+1)
    class_total=[0]*(nclases+1)
    class_aa=[0]*(nclases+1)
    for i in test:
        if(output[i]==0 or truth[i]==0): continue
        total+=1; class_total[truth[i]]+=1
        if(output[i]==truth[i]):
            correct+=1
            class_correct[truth[i]]+=1
    for i in range(1,nclases+1):
        if(class_total[i]!=0): class_aa[i]=100*class_correct[i]/class_total[i]
        else: class_aa[i]=0
        AA+=class_aa[i]
    OA=100*correct/total; AA=AA/nclases_no_vacias

    print('* Accuracy (pixels)')
    for i in range(1,nclases+1): print('  Class %02d: %02.02f'%(i,class_aa[i]))
    print('* Accuracy (pixels) OA=%02.02f, AA=%02.02f'%(OA,AA))
    print('  total:',total,'correct:',correct)

    # guardamos la salida
    if(PAD):
        H1=H1+2*(sizex//2); V1=V1+2*(sizey//2)
        output=np.reshape(output,(-1,H1))
        output=output[sizey//2:V1-sizey//2,sizex//2:H1-sizex//2]

        H1=H1-2*(sizex//2); V1=V1-2*(sizey//2)
        output=np.reshape(output,(H1*V1))
    save_pgm(output,H1,V1,nclases,'../../../Res/output_hvq.pgm')
    # Save the model checkpoint
    torch.save(model.state_dict(),'../../../Mod/model_hvq.ckpt')

    time_end=time.time()

    tb_logger.close()

    print('* Execution time: %.0f s'%(time_end-time_start))
    print('BATCH:',batch_size)
    
    dist.destroy_process_group()

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
    batch_size,
):

    cuda=True if torch.cuda.is_available() else False
    device=torch.device('cuda' if cuda else 'cpu')

    model.training = True
    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, (inputs,labels) in enumerate(train_loader):

        curr_step = start_iter + i
        current_lr = lr_scheduler.get_last_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        height = torch.full((batch_size,), 32, device=device)
        width = torch.full((batch_size,), 32, device=device) 
        
        
        mask = torch.zeros(labels.size(0), 1, 32, 32)
        for k in range(labels.size(0)):
            if labels[k] == 0:
                mask[k] = torch.zeros((1, height[k], width[k]))
            else:
                mask[k] = torch.ones((1, height[k], width[k]))
        
        filenames = [f"patch/{i}_{j}.png" for j in range(labels.size(0))]

        input= {
            "filename": filenames,
            "image": inputs,
            "mask": mask,
            "height": height,
            "width": width,
            "label": labels,
            "clsname": ["hyperspectral"] * labels.size(0),
        }

        outputs = model(input)

        pred=outputs['pred']
        predicted = pred.reshape(pred.size(0), -1).mean(axis=1) #

        loss = outputs['loss']

        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()


def validate(val_loader, model, batch_size):

    cuda=True if torch.cuda.is_available() else False
    device=torch.device('cuda' if cuda else 'cpu')

    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    model.training = False
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    dist.barrier()

    with torch.no_grad():
        for i, (inputs,labels) in enumerate(val_loader):
            # forward
            
            height = torch.full((batch_size,), 32, device=device)
            width = torch.full((batch_size,), 32, device=device)
            
            mask = torch.zeros(labels.size(0), 1, 32, 32)
            for k in range(labels.size(0)):
                if labels[k] == 0:
                    mask[k] = torch.zeros((1, height[k], width[k]))
                else:
                    mask[k] = torch.ones((1, height[k], width[k]))

            filenames = [f"patch/{i}_{j}.png" for j in range(labels.size(0))]
            
            input= {
                "filename": filenames,
                "image": inputs,
                "mask": mask,
                "height": height,
                "width": width,
                "label": labels,
                "clsname": ["hyperspectral"] * labels.size(0),
            }

            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)

            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 100 == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )
    # gather final results
    dist.barrier()
    total_num = torch.Tensor([losses.count]).to(device)
    loss_sum = torch.Tensor([losses.avg * losses.count]).to(device)
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        
        fileinfos, preds, masks, pred_imgs = merge_together(config.evaluator.eval_dir)
        
        shutil.rmtree(config.evaluator.eval_dir)
        # evaluate, log & vis
        ret_metrics, threshold = performances(fileinfos, preds, masks, config.evaluator.metrics)

        log_metrics(ret_metrics, config.evaluator.metrics)

        if args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                pred_imgs,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if args.evaluate and config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
    model.train()
    return ret_metrics, threshold


if __name__ == "__main__":
    main()
