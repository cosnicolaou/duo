
bash run-script.sh gen_ppl_owt_duo --ckpt duo-distilled --steps 32
bash run-script.sh gen_ppl_owt_duo --ckpt duo-distilled --steps 64
bash run-script.sh gen_ppl_owt_duo --ckpt duo-distilled --steps 96
bash run-script.sh eval_owt_duo_distilled
bash run-script.sh zero_shot_duo_distilled
