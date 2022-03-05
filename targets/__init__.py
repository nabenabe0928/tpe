import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow  # noqa: E402

tensorflow.get_logger().setLevel("ERROR")
tensorflow.autograph.set_verbosity(1)
