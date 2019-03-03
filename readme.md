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


            