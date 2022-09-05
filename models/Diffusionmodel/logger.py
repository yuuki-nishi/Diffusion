import makeconfig
import os
class mylogger():
	def __init__(self,config : makeconfig.Myconfig):
		self.filepath = config.loggerpath
		
		if not os.path.exists(config.wavdirpath):
			os.makedirs(config.wavdirpath)
		self.f =open(self.filepath,"w+")

	def write(self,content):
		self.f.write(content)
		return
