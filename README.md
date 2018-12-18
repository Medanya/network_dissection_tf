# network_dissection_tf
Network dissection for tensorflow models, according to http://netdissect.csail.mit.edu/final-network-dissection.pdf

This script runs tensorflow graph to obtain tensors for dissection, then calculates percentiles for each feature maps, after that evaluates interpolated and thresholded features map for segmentation task.

To download mobilenet network and 1 thousand images with segmentation labeling, run:
./download_mobilenet_1k_segmentation.sh 

To install requirments, run: pip2 install -r requirements.txt

To create a demo like this: (link=https://www.dropbox.com/sh/5cazqe58hw38l1w/AABN1LAdIlydlK0HjH__H5ula?dl=0) run:
python2 run_dissection.py
