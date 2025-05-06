

for ckpt in /cs224u/cache/duo-checkpoints/duo.ckpt /cs224u/cache/duo-checkpoints/duo-distilled.ckpt /cs224u/duo/outputs/sentiment/2025.05.01/213454/checkpoints/best.ckpt ; do
    echo "Testing $ckpt"
    bash run-script.sh test_sentiment_finetuned_duo.sh --steps 32 --seed 1 --ckpt $ckpt 
done