
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from typing import List
import os
from . import sounddata

def makeclasslist(x:str) -> List[int]:
  ret = [0 for i in range(sounddata.classnum)]
  name=os.path.basename(x)[0:3]
  #print(x)
  if not name in sounddata.personid:
    name=os.path.basename(x)
  ret[sounddata.classdict[name]]=1
  #print(len(ret))
  return ret
print(sounddata.classnum)
print(len([ makeclasslist(sounddata.soundclass[i]) for i in range(sounddata.classnum) ]))#ok
values=tf.constant([ makeclasslist(sounddata.soundclass[i]) for i in range(sounddata.classnum) ])
print(values.shape)
def getlabelfrompath(path):
  #print(path)
  if path=="0" or path == "zeros":
    return "zeros"
  name=os.path.basename(path)
  if name[0:3] in sounddata.personid:
    return name[0:3]
  else:
    try:
      for k in sounddata.noiseclass:
        if k in name:
          return k#名前の中にノイズの種類がある場合
      raise ValueError("no such class {}".format(name))
    except ValueError as e:
      print(e)

# build a lookup table
"""table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(soundclass),
        values=values,
        key_dtype=tf.string,
        value_dtype=tf.int32
    ),
    default_value=tf.constant(-1)
)
def getfromtable(x):
  print(table.key_dtype)
  return table.lookup(x)"""