import resolve_lib
import sys
import argparser
import makeconfig
import ClassifierModel
import GaussianModel
import dataset as datamake
import torch
import torch.nn as nn
import metrics
import os
import wavsamplewriter
from torchsummary import summary
import graphwriter
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
#import logger as log
MASTER_NUMBER = 0
def main(rank,world_size,port):
	torch.backends.cudnn.benchmark = True
	avail_cuda_device = [0,2]
	device = torch.device('cuda', avail_cuda_device[rank])
	torch.cuda.set_device(device)
	args = argparser.argparse()
	config = makeconfig.makeconfig(args)
	config.batch_size = config.batch_size // world_size
	config.setdevice(device)
	#logger = log.mylogger(config)
	#stdoutpath = config.stdoutpath
	#sys.stdout = open(stdoutpath, 'w')#標準出力の変更
	
	# create default process group
	
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = str(port)#ポートに注意
	
	torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
	replica_count = torch.cuda.device_count()

	Classifier = ClassifierModel.Classifier(config)
	Gausemodel = GaussianModel.DiffusionModel(config,Classifier,device).to(device)
	DiffusiomModel = torch.nn.parallel.DistributedDataParallel(Gausemodel, device_ids=[device])
	#DiffusiomModel =GaussianModel.DiffusionModel(config,Classifier,device)
	optimizer = torch.optim.Adam(DiffusiomModel.parameters(), lr=2e-4)
	
	# 学習のループ
	losses = []
	stdes = []
	start_epoch = 0
	#checkpointから再開する場合
	#load_checkpoint = True
	ckptpath = config.checkpointpath+"/ckpt.pt"
	#ckptpath = "/net/shard"+"/work/y-nishi/Diffwave/diffwave/pretrained/weights.pt"#配布されていたやつ
	scaler =torch.cuda.amp.GradScaler()
	autocast = torch.cuda.amp.autocast()
	if os.path.exists(ckptpath):
		cptfile = ckptpath
		cpt = torch.load(cptfile)
		stdict_m = cpt['model_state_dict']
		#stdict_m = cpt['model']
		stdict_o = cpt['opt_state_dict']
		stdes = cpt['std']
		start_epoch = cpt['epoch'] + 1#startしたいepochだからここで+1
		#DiffusiomModel.module.DiffSE.load_state_dict(stdict_m)
		optimizer.load_state_dict(stdict_o)
		losses = cpt['loss']
		scaler.load_state_dict(cpt['scaler'])
	#GPUに送る
	#DiffusiomModel = DiffusiomModel.to(device)
	#summary(DiffusiomModel,(3,1,80000))
	#multi gpu learning
	#DiffusiomModel = torch.nn.DataParallel(DiffusiomModel)
	print("data dir")
	print(config.train_data_dir)
	dataset = datamake.MyDataset(config,path=config.train_data_dir)

	print("device number : {}".format(dist.get_rank()))
	print("datanum : {}".format(len(dataset)))
	train_sampler = torch.utils.data.distributed.DistributedSampler(
						dataset, 
						#num_replicas=dist.get_world_size(), 
						#rank=dist.get_rank(),
						shuffle=True,)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, 
					#shuffle=False,  
					sampler=train_sampler,
					pin_memory = True,
					drop_last=True,
					num_workers=1
					)
	#dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=3,shuffle = True)
	melconverter = GaussianModel.Melconvert(config)
	#start training
	print(torch.cuda.get_device_name(0))
	#print(torch.cuda.get_device_name(1))#こういう風にするのは意味ないっぽい
	print('Memory Usage:')
	print('Allocated:', torch.cuda.memory_allocated(device=device)/1024**3, 'GB')
	print('Cached:   ', torch.cuda.memory_cached(device=device)/1024**3, 'GB')
	print("avail device count : {}".format(torch.cuda.device_count()))
	#もしcheckpointpathが無かったら作る
	if not os.path.exists(config.checkpointpath):
		os.makedirs(config.checkpointpath)
	# スケーラーを定義
	torch.manual_seed(114514)
	train_num = 0
	lossfunc = nn.L1Loss()
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)
	print("start epoch : {}".format(start_epoch))
	for epoch in range(start_epoch,50):
		iter_num = 0
		itrcount = 0
		break
		for data,melspec,soundclass in dataloader:
			optimizer.zero_grad()
			train_num += 1
			itrcount +=1
			nowbatchsize = data.size()[0]
			if nowbatchsize != config.batch_size:
				break
			#GPUに送る
			#print("audio size")
			#print(data.size())
			data = data.to(device)
			data = data.reshape([config.batch_size,1,config.hopsize_quant * config.crop_mel_frames])
			#noised_data = DiffusiomModel.module.get_noised_signal(data).reshape([config.batch_size,1,config.samplenum]).float()
			#wavsamplewriter.wavwriter(noised_data[0].float().cpu(),"in_itr_test",config)#大丈夫だった
			#melspec = melconverter.melspec(data.cpu())
			#print("spec1")
			#print(melspec[0])
			#print("spec2")
			#print(melspec[0])

			melspec = melspec.to(device).squeeze(1)
			assert melspec.shape ==(nowbatchsize,config.n_mels,config.crop_mel_frames )
			with autocast: 
				estimated,noise,step = DiffusiomModel(data,melspec)#入力となるx_0はclean音声であることに注意
			l1loss=lossfunc(estimated,noise)
			
			noise = noise.squeeze(1)#[B,1,L]のサイズだったので
			estimated = estimated.squeeze(1)
			#print(cos(noise,estimated))
			#print(noise.size())
			cosloss =1- cos(noise,estimated).mean()
			loss =l1loss+ cosloss*0.1
			itemed_loss=loss.item()
			if rank == MASTER_NUMBER:
				print('Memory Usage after calc:')
				print('Allocated:', torch.cuda.memory_allocated(device=device)/1024**3, 'GB')
				print('Cached:   ', torch.cuda.memory_cached(device=device)/1024**3, 'GB')
				smean = (step*1.0).mean()
				print("epoch : {}, iter : {}, loss is {},step_mean : {}, l1loss is {}".format(epoch,iter_num,itemed_loss,smean,l1loss))
				print("estimated std : {}".format(estimated.std()))
				cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
				print("cos : {}".format(cos(noise,estimated).mean()))
				#logger.write("epoch : {}, iter : {},  loss is {}, ".format(epoch,iter_num,itemed_loss))
			#with torch.no_grad():
			#	estimated_data = DiffusiomModel.module.reverse(data,melspec)
			#loss.backward()
			stdes.append(estimated.std().item())
			losses.append(itemed_loss)#loss.item()で、python float として保存
			#print(estimated)
			#print("estimated :  mean ; {}, std : {}".format(torch.mean(estimated),estimated.std()))

			#optimizer.step()
			#metrics=metrics.metrics(data,estimated_data,None)
			# スケールした勾配を作る
			scaler.scale(loss).backward() 
			scaler.unscale_(optimizer)
			grad_norm = nn.utils.clip_grad_norm_(DiffusiomModel.parameters(), 1e9)

			# 勾配をアンスケールしてパラメータの更新
			scaler.step(optimizer) 

			# スケーラーの更新
			scaler.update() 
			iter_num += 1
			#print("itrcount : {}".format(itrcount))
			del loss
		print("itrcount : {}".format(itrcount))
		# 学習情報の保存
		if rank == MASTER_NUMBER:
			torch.save({'epoch': epoch,
					'model_state_dict': DiffusiomModel.module.state_dict(),
					'opt_state_dict': optimizer.state_dict(),
					'loss': losses,
					'std': stdes,
					'scaler':scaler.state_dict()
					}, config.checkpointpath+"/ckpt.pt")
	#write sample
	#evaldataset ,_= next(iter( torch.utils.data.DataLoader(datamake.MyDataset(config,path=config.eval_data_dir))))
	#evaldataset ,_= next(iter( torch.utils.data.DataLoader(datamake.MyDataset(config,path=config.eval_data_dir))))
	sampleidx = [0,10,100,300]
	if rank == MASTER_NUMBER:
		for samplei in sampleidx:
			evalsample,spec,_ = dataset[samplei]
			#noise schedule がokかのテスト
			
			#with torch.no_grad():
			#	for i in range(config.stepnum+1):
			#		noised = DiffusiomModel.module.get_tth_noised_signal(evalsample.unsqueeze(0).to(device),i)
					
			#print(noised.size())
			#wavsamplewriter.wavwriter(noised[0].float().cpu(),"n_{}".format(i),config)
			wavsamplewriter.wavwriter(evalsample,"clean_{}".format(samplei),config)
			noised = torch.randn_like(torch.from_numpy( evalsample)).unsqueeze(0)
			spec = spec.unsqueeze(0).to(device)
			wavsamplewriter.wavwriter(noised[0].float().cpu(),"noised_{}".format(samplei),config)
			noised = noised.unsqueeze(0).to(device)
			#estimated_melspec = melconverter.melspec(evalsample.float().cpu()).to(device)
			with torch.no_grad():
				estimated_datas = DiffusiomModel.module.reverse(noised.to(device),spec).squeeze()#denoiseの履歴を返すので
				estimated_data = estimated_datas[0]#denoiseの履歴を返すので
			
			for i in range(0,config.stepnum+1):
				wavsamplewriter.wavwriter(estimated_datas[i].float().cpu(),"estimated_{}_{}".format(samplei,i),config)
				
			wavsamplewriter.wavwriter(estimated_data.float().cpu(),"estimated_{}".format(samplei),config)
		graphwriter.write(losses,stdes,config)
	return

def _get_free_port():
	import socketserver
	with socketserver.TCPServer(('localhost', 0), None) as s:
		return s.server_address[1]

if __name__ == '__main__':
	print("GPU STATE")
	print(torch.cuda.is_available())
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("device is : {}".format(device))
	# initialization mode.
	port = _get_free_port()
	
	#world_size = torch.cuda.device_count()
	world_size = 2
	multiprocessing.spawn(main,
		args=(world_size,port),
		nprocs=world_size,
		join=True)

	#main(device)