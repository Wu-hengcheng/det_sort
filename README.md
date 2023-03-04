# person_det

for deploying in beijing 20220708


# update
20230223: add deepsort code to detection code, now program can detect and track.    
20230224: the real detect and track deploy code. now it can run on youkong.    
20230225:  optimize  label to show class names ,  add different colors to boxes , internal convert engine function

# todo

# how to use
1. if first run on device , write deepsort.onnx path in config.yaml , then change to deepsort.engine path
2. cd det_sort/build and build the project
3. if first time run , use command:  ./perception -c 
4. after first run , just execute ./perception