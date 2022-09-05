# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Makes lists of background and foreground source files."""
import argparse
import glob
import os
import soundfile as sf


def getclass(path):
  return os.path.basename(path)[0:3]

def argmaxclass(fsd_dir):
  subsets = ['train', 'validation', 'eval']
  classes = set()
  for subset in subsets:
    file_list = glob.glob(os.path.join(fsd_dir, subset, '*', '*.wav'))
    for file in file_list:
      classes.add(getclass(file))
  dictclass = {}
  for c in classes:
    dictclass[c]=0
  for subset in subsets:
    file_list = glob.glob(os.path.join(fsd_dir, subset, '*', '*.wav'))
    for file in file_list:
      dictclass[getclass(file)]+=1
  items_sorted = [ i[0] for i in sorted(dictclass.items(), key=lambda item: item[1], reverse=True)]
  bestclass=items_sorted[0:10]
  print( sorted(dictclass.items(), reverse=True))
  print(dictclass[bestclass[0]])
  print(dictclass[bestclass[1]])
  print(dictclass[bestclass[2]])
  print(items_sorted)
  for i in items_sorted:
    print("{} : {}".format(i,dictclass[i]))
  return bestclass
  
def fileterclass(file_list,bestclass):
  return [i for i in file_list if getclass(i) in bestclass]

def make_lists(fsd_dir,bestclass):
  """Makes background and foreground source lists under fsd_dir subsets."""
  subsets = ['train', 'validation', 'eval']
  short_cutoff = 10.0
  print(bestclass)
  for subset in subsets:
    long_file_list = []
    short_file_list = []

    file_list = glob.glob(os.path.join(fsd_dir, subset, '*', '*.wav'))
    file_list = fileterclass(file_list,bestclass)
    #print(file_list)
    for myfile in file_list:
      relative_path = os.path.relpath(myfile, fsd_dir)

      file_info = sf.info(myfile)

      if file_info.duration > short_cutoff:
        long_file_list.append(relative_path)
      else:
        short_file_list.append(relative_path)

    n_short = len(short_file_list)
    if n_short > 0:
      list_name = os.path.join(fsd_dir, subset + '_foreground_sound.txt')
      with open(list_name, 'w') as f:
        f.writelines('\n'.join(short_file_list))
      print('Generated foreground file list of {} files '
            'for {}.'.format(n_short, subset))

    n_long = len(long_file_list)
    if n_long > 0:
      list_name = os.path.join(fsd_dir, subset + '_background_sound.txt')
      with open(list_name, 'w') as f:
        f.writelines('\n'.join(long_file_list))
      print('Generated background file list of {} files '
            'for {}.'.format(n_long, subset))


def main():
  parser = argparse.ArgumentParser(
      description='Makes background and foreground source lists.')
  parser.add_argument(
      '-d', '--data_dir', help='FSD data main directory.', required=True)
  args = parser.parse_args()
  bestclass=argmaxclass(args.data_dir)
  make_lists(args.data_dir,bestclass)

if __name__ == '__main__':
  main()
