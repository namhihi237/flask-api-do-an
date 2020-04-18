import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import json

n_layers = 24
n_neurons = 6
n_inputs = 4
n_outputs = 1

app = Flask(__name__)


@app.route('/')
def hi():
    print('hi')
    return 'hello'


@app.route('/api/v1', methods=['POST'])
def apiv1():
    # get data from request
    data = request.get_json()
    data = data['input']
    # handle data
    input_data = np.empty((0, 4))
    for item in data:
        list_values = [v for v in item.values()]
        x = np.array(list_values, dtype=np.float32).reshape(-1, 4)
        input_data = np.concatenate((input_data, x), axis=0)

    X0 = input_data
    print(X0)
    try:
        X_batches = X0.reshape(-1, n_layers, n_inputs)
        y_pred = sess.run(get, feed_dict={X: X_batches})
        print(y_pred)
        return {"result": str(y_pred[-1][-1][-1])}
    except:
        print("error shape input")
        return 'error'


if __name__ == '__main__':
    # load model and run server
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.import_meta_graph("models/model.ckpt.meta")
        save_path = saver.restore(sess, "models/model.ckpt")
        graph = tf.compat.v1.get_default_graph()
        get = graph.get_tensor_by_name('Reshape_1:0')
        X = graph.get_tensor_by_name('Placeholder:0')
        app.run(debug=True)
