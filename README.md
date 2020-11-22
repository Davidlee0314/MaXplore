# MaXplore

This is an CNN testing inputs generation technique, implemented based on [DeepXplore](https://github.com/peikexin9/deepxplore). I substituted the original *Neuron Coverage* into *Strong Neuron Activation Coverage* proposed in [DeepGauge](https://deepgauge.github.io/).

## Run with Docker

1. Build up a docker image.
```
$ docker build -t maxplore
```

2. Run the image and mount the output to the local directory.
```
$ docker run --mount type=bind,source="$(pwd)"/,target=/app maxplore
```
3. To run each experiment separately, just comment out corresponding lines in `all_exp.py` and rebuild the image.

## Run with python virtualenv

1. Create a python virtual environment and activate it.
2. Install required dependencies.
```
$(venv) pip install -r requirements.txt
```
3. Run the all_exp.py.
```
$(venv) python3 all_exp.py
```
4. To run each experiment separately, just comment out corresponding lines in `all_exp.py`.

---
**NOTE**

The 3rd experiments run about 30 minutes. To avoid using **MaXplore** on 16 filters in total, change `one_filter = True` in `MaXplore/configs.py`

---

## Output

### - gen_test_exp*/
These directories contrain newly created MNIST image test inputs and orginal images.

### - exp*_plot.jpg
Box plots for comparing in terms of *Strong Neuron Activation Coverage (SNACov)*.

### - max_stat_exp*.txt
Output logs for each experiment and layers respective *SNACov*.

### - deepxplore.txt
The *SNACov* output with DeepXplore. Since it's not using eager execution in tensorflow, I decide to output it separately. To rerun it, you have to create a virtual environment for this project.
```
$(venv) python3 DeepXplore/gen_diff.py
```