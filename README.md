# my-nerf
This is my CSE 291 final project, implementing NeRF on the bottles dataset. The code is implemented with reference to https://github.com/airalcorn2/pytorch-nerf.

To train, edit the hyperparmeters in `run_nerf.py` and run it. After training is complete, run `predict.py` to generate the results in `result` folder, which are the rgb map and depth map of five novel views. The folder already contains the result that I got. To attain those results, I trained the model for about one day on an RTX 3060.
