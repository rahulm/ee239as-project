# ee239as-project

## Comparison of face generation methods and techniques.


"configurations": [
        {
            "name": "Python: Current File (Integrated Terminal)",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "args": [
                "--decoder", "naive",
                "--encoder", "naive",
                "--epochs", "50",
                "--batch_size", "128",
                "--intermediate_dim", "512",
                "--latent_dim", "2",
                "--mode", "train",
                "--res", "28",
                "--loss", "ce",
                // // "seed", "",
                // // "device", "",
                // // "image_dir", "",
                // // "landmark_dir", "",
                // // "male_img_dir", "",
                // // "female_img_dir", "",
                // // "male_landmark", "",
                // // "female_landmark", "",
                // "path", "tpath",
                // "log", "tlogs",
                // "appear_lr", "7e-4",
                // "landmark_lr", "7e-4",
            ],




To run code:
MOBILENET: 
Note: don't add `--use_subset` if you want to train on full dataset
```
python train.py --decoder naive --encoder naive --epochs 50 --batch_size 16 --intermediate_dim 512 --latent_dim 50 --mode train --img_width 256 --img_height 256 --loss mse --name celeb --dataset /home/odin/Downloads/Celeb --use_subset
```

