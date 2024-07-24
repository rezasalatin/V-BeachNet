python3 train_video_seg.py \
    --gpu=0 \
    --dataset=Training_Station/Duck_Rectified/ \
    --seed=-1 \
    --log \
    --lr=1e-5 \
    --lu=0.5 \
    --scheduler-step=25 \
    --total-epochs=100 \
    --budget=300000 \
    --obj-n=2 \
    --clip-n=6 \
	--new
#    --resume=logs/bestmodel/model/best.pth
# activate new or resume
