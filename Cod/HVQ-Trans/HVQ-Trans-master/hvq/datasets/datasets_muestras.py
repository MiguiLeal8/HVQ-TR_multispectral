#-----------------------------------------------------------------
# FUNCIONES PARA LEER DATASETS Y SELECCIONAR MUESTRAS
#-----------------------------------------------------------------

import math,random,struct,os,time,sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn import preprocessing
import torchvision.transforms as transforms
from sklearn import metrics

from PIL import Image
import matplotlib.pyplot as plt

AUM=0  # aumentado: 0-sin_aumentado, 1-con_aumentado

def save_anomaly_detection_as_rgb_image(preds, labels, H, V, sizex=32, sizey=32, filename="../../../Res/anomaly_detection_rgb.png"):       
  
  preds=preds.reshape(H,V)
  labels=labels.reshape(H,V)

  class_colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)]

  anomaly_detection_color = np.zeros_like(preds, dtype=np.uint8)
  
  anomaly_detection_color[labels == -1] = 0                  # Pixel que no corresponde a la imagen (negro)
  anomaly_detection_color[(preds == 0) & (labels == 0)] = 0  # Correctamente clasificado como normal (negro)
  anomaly_detection_color[(preds != 0) & (labels == 0)] = 3  # Falso positivo (azul)
  anomaly_detection_color[(preds == 0) & (labels == 1)] = 2  # Falso negativo (rojo)
  anomaly_detection_color[(preds != 0) & (labels == 1)] = 1  # Correctamente clasificado como anomalía (verde)

  colored_image = np.zeros((H, V, 3), dtype=np.uint8)
  for class_index, color in enumerate(class_colors):
    colored_image[anomaly_detection_color == class_index] = color

  # Create colored image
  norm_img = Image.fromarray(colored_image).convert("RGB")
  norm_img.save(filename)
  print("* Saved anomaly_detection_rgb.png")
 
  labels_flat = labels.flatten()
  preds_flat = preds.flatten()

  vp = np.sum((preds != 0) & (labels == 1))
  rvp = 100 * vp / len(labels_flat)
  vn = np.sum((preds == 0) & (labels == 0))
  rvn = 100 * vn / len(labels_flat)
  fp = np.sum((preds != 0) & (labels == 0))
  rfp = 100 * fp / len(labels_flat)
  fn = np.sum((preds == 0) & (labels == 1))
  rfn = 100 * fn / len(labels_flat)

  # Calcula la curva ROC
  fpr, tpr, thresholds = metrics.roc_curve(labels_flat, preds_flat, pos_label=1)

  # Calcula el AUC
  auc = metrics.auc(fpr, tpr)

  print("VP = ", vp, " -> ", rvp, "%")
  print("VN = ", vn, " -> ", rvn, "%")
  print("FP = ", fp, " -> ", rfp, "%")
  print("FN = ", fn, " -> ", rfn, "%")
  print("AUC = ", auc)

def save_grey_scale_image(pred_mean, filename="../../../Res/grey_scale.png"):
  top = np.sort(pred_mean.flatten())[-500:]
  bottom = np.sort(pred_mean.flatten())[:500]
  
  # Normaliza los valores al rango [0, 255]
  norm_array = (pred_mean - pred_mean.min()) / (pred_mean.max() - pred_mean.min()) * 255.0
  norm_array = norm_array.astype(np.uint8)

  # Convierte a imagen en escala de grises
  grayscale_img = Image.fromarray(norm_array, mode='L')
    
  # Guarda la imagen
  grayscale_img.save(filename)
  print("* Saved grey_scale.png")

def show_hist(pred_mean):
  pred_mean = pred_mean.flatten()
  plt.figure(figsize=(8, 5))
  plt.hist(pred_mean, bins=100, color='skyblue', edgecolor='black')
  plt.title("Histograma de pred_mean")
  plt.xlabel("Valor de predicción promedio")
  plt.ylabel("Frecuencia")
  plt.grid(True)
  plt.show()

def read_raw(fichero):
  (B,H,V)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)
  print('* Read dataset:',fichero)
  print('  B:',B,'H:',H,'V:',V)
  print('  Read:',len(datos))
  
  datos=datos.reshape(V,H,B)
  datos=torch.FloatTensor(datos)
  return(datos,H,V,B)

def save_raw(output,H,V,B,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    f.write(struct.pack('i',B))
    f.write(struct.pack('i',H))
    f.write(struct.pack('i',V))
    output=output.reshape(H*V*B)
    for i in range(H*V*B):
      f.write(struct.pack('i',np.int(output[i])))
    f.close()
    print('* Saved file:',filename)

def read_pgm(fichero):
  try:
    pgmf=open(fichero,"rb")
  except IOError:
    print('No puedo abrir ',fichero)
  else:
    assert pgmf.readline().decode()=='P5\n'
    line=pgmf.readline().decode()
    while(line[0]=='#'):
      line=pgmf.readline().decode()
    (H,V)=line.split()
    H=int(H); V=int(V)
    depth=int(pgmf.readline().decode())
    assert depth<=255
    raster=[]
    for i in range(H*V):
      raster.append(ord(pgmf.read(1)))
    print('* Read GT:',fichero)
    print('  H:',H,'V:',V,'depth:',depth)
    print('  Read:',len(raster))
    return(raster,H,V)

def save_pgm(output,H,V,nclases,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    cadena='P5\n'+str(H)+' '+str(V)+'\n'+str(nclases)+'\n'
    f.write(bytes(cadena,'utf-8'))
    f.write(output)
    f.close()
    print('* Saved file:',filename)


def select_training_samples(truth,H,V,sizex,sizey,porcentaje):
  print('* Select training samples')
  # hacemos una lista con las clases, pero puede haber clases vacias
  nclases=0; nclases_no_vacias=0
  N=len(truth)
  for i in truth:
    if(i>nclases): nclases=i
  print('  nclasses:',nclases)
  lista=[0]*nclases
  for i in range(nclases):
    lista[i]=[]
  for i in range(int(sizey/2),V-int(sizey/2)):
    for j in range(int(sizex/2),H-int(sizex/2)):
      ind=i*H+j
      if(truth[ind]>0): lista[truth[ind]-1].append(ind)
  for i in range(nclases):
    random.shuffle(lista[i]) 
  # seleccionamos muestras para train, validacion y test
  print('  Class  # :   total | train |   val |    test')
  train=[]; val=[]; test=[]
  test_index=[] 
  for i in range(nclases):
    # tot0: numero muestras entrenamiento, tot1: validacion 
    if(porcentaje[0]>=1): tot0=porcentaje[0]
    else: tot0=int(porcentaje[0]*len(lista[i]))
  
    if(tot0>=len(lista[i])): tot0=len(lista[i])//2
    if(tot0<0 and len(lista[i])>0): tot0=1
    if(tot0!=0): nclases_no_vacias+=1

    if(porcentaje[1]>=1): tot1=porcentaje[1]
    else: tot1=int(porcentaje[1]*len(lista[i]))

    if(tot1>=len(lista[i])-tot0): tot1=(len(lista[i])-tot0)//2
    if(tot1<1 and len(lista[i])>0): tot1=0
    
    if(i==1): tot0=0 # entrenar solo con normales

    for j in range(len(lista[i])):
      if(j<tot0): train.append(lista[i][j])
      elif(j<tot0+tot1): val.append(lista[i][j])
       
      test.append(lista[i][j])
    print('  Class',f'{i+1:2d}',':',f'{len(lista[i]):7d}','|',f'{tot0:5d}','|',
      f'{tot1:5d}','|',f'{len(lista[i])-tot0-tot1:7d}')
  return(train,val,test,nclases,nclases_no_vacias)

def select_patch(datos,sizex,sizey,x,y):
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2));     
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2));
  patch=datos[:,y1:y2,x1:x2]
  return(patch)


#-----------------------------------------------------------------
# PYTORCH - SETS
#-----------------------------------------------------------------

# cogemos muestras sin ground-truth (dadas por el indice samples)
class HyperAllDataset(Dataset):
  def __init__(self,datos,samples,H,V,sizex,sizey):
    self.datos=datos; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose(
      [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; H=self.H; V=self.V; B=self.B
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    return(patch)

#----------------

# cogemos muestras con ground-truth (dadas por el indice samples)
class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey):
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose(
      [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; truth=self.truth; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    # renumeramos porque la red clasifica tambien la clase 0
    return(patch,truth[self.samples[idx]]-1) # truth pasa de tener valores 2 y 1 a 1 y 0

# For updating learning rate manual
def update_lr(optimizer,lr):    
  for param_group in optimizer.param_groups:
    param_group['lr']=lr

# calcula los promedios de precisiones
def accuracy_mean_deviation(OA,AA,aa):
  n=len(OA); nclases=len(aa[0])
  print('* Means and deviations (%d exp):'%(n))
  # medias
  OAm=0; AAm=0; aam=[0]*nclases;
  for i in range(n):
     OAm+=OA[i]; AAm+=AA[i]
     for j in range(1,nclases): aam[j]+=aa[i][j]
  OAm/=n; AAm/=n
  for j in range(1,nclases): aam[j]/=n
  # desviaciones, usamos la formula que divide entre (n-1)
  OAd=0; AAd=0; aad=[0]*nclases
  for i in range(n):
     OAd+=(OA[i]-OAm)*(OA[i]-OAm); AAd+=(AA[i]-AAm)*(AA[i]-OAm)
     for j in range(1,nclases): aad[j]+=(aa[i][j]-aam[j])*(aa[i][j]-aam[j])
  OAd=math.sqrt(OAd/(n-1)); AAd=math.sqrt(AAd/(n-1))
  for j in range(1,nclases): aad[j]=math.sqrt(aad[j]/(n-1))
  for j in range(1,nclases): print('  Class %02d: %02.02f+%02.02f'%(j,aam[j],aad[j]))
  print('  OA=%02.02f+%02.02f, AA=%02.02f+%02.02f'%(OAm,OAd,AAm,AAd))
