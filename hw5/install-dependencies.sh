#Clone and make the darknet repository

echo "Cloning GitHub Repo and downloading required model weights"

git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make

wget https://pjreddie.com/media/files/yolov2-voc.weights
wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights

echo "Running inference and saving as a textfile"

./darknet detector test cfg/voc.data cfg/yolov2-voc.cfg yolov2-voc.weights -dont_show -ext_output < ../valid.txt  > ../standard_results.txt

./darknet detector test cfg/voc.data cfg/yolov2-tiny-voc.cfg yolov2-tiny-voc.weights -dont_show -ext_output < ../valid.txt  > ../tiny_results.txt

echo "Process Completed"