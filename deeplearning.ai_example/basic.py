import numpy as np
import tensorflow as tf

input_x = np.array([[1.0], [-20.], [100.]])

w = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32, [3, 1])

#cost=tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
#cost = w**2 - 10*w + 25
cost = x[0][0] * w**2 + x[1][0] * w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
for i in range(1000):
    session.run(train, feed_dict={x: input_x})
print(session.run(w))


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    ### START CODE HERE ###

    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(tf.float32, name="logits")
    y = tf.placeholder(tf.float32, name="labels")

    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()

    # Run the session (approx. 1 line).
    cost = sess.run(cost, feed_dict={z: logits, y: labels})

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return cost


def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """

    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name='x')

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above.
    # You should use a feed_dict to pass z's value to x.
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict={x: z})

    ### END CODE HERE ###

    return result


if __name__ == "__main__":
    # logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
    # cost = cost(logits, np.array([0, 0, 1, 1]))
    # print("cost = " + str(cost))
    # logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
    # cost = cost(logits, np.array([0, 0, 1, 1]))
    # print("cost = " + str(cost))

    indices = [0, 2, -1, 1]
    depth = 4
    axis = -1
    one_hot = tf.one_hot(indices, 3, axis=-1)

    print("**********\n")
    with tf.Session() as sess:
        print(sess.run(one_hot))

    a = tf.ones([3,5])

    sess = tf.Session()
    w = tf.get_variable('w1',[3,5], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    # a = sess.run(a)
    w = sess.run(w)
    print(w)
    sess.close()

    tf.get_variable('W1',[23,56], initializer= tf.contrib.layers.xavier_initialzier(seed = 2))

    initializer = tf.zeros_initializer()

    parameters = {
        'w1':w1,
        'dla': 3242,
        'nieg':43
    }
    tf.nn.tanh()

    tf.reduce_mean

    tf.nn.softmax_cross_entropy_with_logits(Z2,Y)

    tf.nn.softmax_cross_entropy_with_logits()

_, c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
tf.train.AdamOptimizer()

_, c = sess.run([optimizer, cost],feed_dict = {X:})

    tf.equal(tf.argmax(Z3), tf.argmax(Y))

    tf.truncated_normal


    
