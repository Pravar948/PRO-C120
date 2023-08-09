import nltk
import tensorflow as tf
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
embed = hub.load(module_url)

pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How can I help you today?"]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there"]
    ],
    [
        r"what is your name ?",
        ["I am a customer support bot, you can call me whatever you like"]
    ],
    [
        r"how can i help you ?",
        ["I am here to help you with any questions or concerns you may have. What can I assist you with today?"]
    ],
    [
        r"sorry (.*)",
        ["Its alright","Its OK, never mind"]
    ],
    [
        r"i am looking for (.*)",
        ["Why don't you tell me more about what you're looking for?"]
    ],
    [
        r"quit",
        ["Thank you for contacting us. Have a great day!"]
    ],
]

def customer_support_bot(sentence):
    embeddings = embed([sentence])["outputs"]
    similarity_input_placeholder = tf.placeholder(tf.float32, shape=(None, 512))
    similarity_message_encodings = tf.nn.l2_normalize(embeddings, axis=1)
    similarity = tf.matmul(similarity_input_placeholder, similarity_message_encodings, transpose_b=True)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        response = session.run(similarity, feed_dict={similarity_input_placeholder: embeddings})
        return response

chatbot = Chat(pairs, customer_support_bot)
chatbot.converse()