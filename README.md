# Example to use TensorBoard from PyTorch 

This repository shows how to use TensorBoard from PyTorch with MNIST example. 

[tensorboardX](https://github.com/lanpa/tensorboard-pytorch) module, a TensorBoard extension for PyTorch, is used for this.

## Usage

```bash
pip install -r requirements.txt
python main.py # start streaming data to the TensorBoard while training the model
```

```bash
# In another terminal, 
tensorboard --logdir runs

# Then open localhost:6006 in your browser, and you will see something like the screenshots below.
```

## Screenshots

### Loss and accuracy

<img src="screenshots/scalars_loss_accuracy.png" />

### Model architecture

<img src="screenshots/graphs_model.png" />

### Model parameters

<img src="screenshots/histograms_params.png" />

## License

[MIT License](LICENSE.txt)
