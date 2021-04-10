# StyleGestures-WithSmallMemory

Repair OOM(out of Memory) in StyleGestures

The original repo: https://github.com/simonalexanderson/StyleGestures/

Overall I did three things to advoid OOM

## 1 Split  files into pieces

Before traing, I split big training file into small pieces.

You should replace the "/StyleGestures-master/motion/datasets/locomotion.py" with locomotion-new.py

## 2 Move the data-transform function to getitem

Honestly, it takes  a lot of  time to acomplish this because the data-transform process of StyleGestures is super-complicated.

You should replace the "/StyleGestures-master/motion/datasets/motion_data.py" with motion_data-new.py

## 3 Remove some redundant "data.copy()"

I found some data.copy() is unnecessary， but some is necessary. It depends on whether the data will be reused later. If not, the data.copy() is unnecessary.
