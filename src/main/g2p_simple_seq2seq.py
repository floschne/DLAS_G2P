import logging
import os
import sys
import time

from sklearn.model_selection import train_test_split
import data_util as dus
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

def prepare_data(data_path,
                 go_token='<GO>', add_go_token=True,
                 pad_token='<PAD>', apply_padding=False,
                 end_token='<END>', add_end_token=True,
                 use_numpy_arrays=True):

    # load the cmudict data
    du = dus.DataUtil(data_path)
    grapheme_vocab, phoneme_vocab, grapheme_sequences, phoneme_sequences, max_grapheme_seq_len, max_phoneme_seq_len = du.get_sequential_cmudict_for_lstm(
        stress_indicators=False)

    # encode the graphemes and phonemes vocabularies to numerical values
    grapheme_vocab_encoder = dus.SimpleVocabularyEncoder(grapheme_vocab)
    phoneme_vocab_encoder = dus.SimpleVocabularyEncoder(phoneme_vocab)

    # add the pad_token to grapheme vocab
    if apply_padding:
        grapheme_vocab_encoder.add_vocab_symbol(pad_token)

    # add the go_token to the phoneme vocab
    if add_go_token:
        phoneme_vocab_encoder.add_vocab_symbol(go_token)

    # add the end_token to the phoneme vocab
    if add_end_token:
        phoneme_vocab_encoder.add_vocab_symbol(end_token)

    grapheme_vocab_size = grapheme_vocab_encoder.get_num_classes()
    phoneme_vocab_size = phoneme_vocab_encoder.get_num_classes()

    # encode all grapheme and phoneme sentences
    encoded_graphemes = []
    encoded_phonemes = []
    for g_seq in grapheme_sequences:
        g_seq_ = g_seq
        if apply_padding:
            # insert pad_token at front so that each grapheme seq has the same length
            g_seq_ = ([pad_token] * (max_grapheme_seq_len - len(g_seq))) + g_seq
        if use_numpy_arrays:
            g_seq_ = np.array(g_seq_)
        encoded_graphemes.append(grapheme_vocab_encoder.encode_sequence(g_seq_))
    for p_seq in phoneme_sequences:
        p_seq_ = p_seq
        # add the go token to the phoneme sequences
        if add_go_token:
            p_seq_ = ([go_token] + p_seq)
        # add the go token to the phoneme sequences
        if add_end_token:
            p_seq_ = (p_seq_ + [end_token])
        if use_numpy_arrays:
            p_seq_ = np.array(p_seq_)
        encoded_phonemes.append(phoneme_vocab_encoder.encode_sequence(p_seq_))

    if use_numpy_arrays:
        encoded_graphemes = np.array(encoded_graphemes)
        encoded_phonemes = np.array(encoded_phonemes)

    return grapheme_vocab_size, phoneme_vocab_size, grapheme_vocab_encoder, phoneme_vocab_encoder, encoded_graphemes, encoded_phonemes


def build_graph(gra_vocab_size, pho_vocab_size, embedding_dim=10, num_hidden_units=50, learning_rate=0.001,
                use_dropout=False, keep_prob=.8):
    # Placeholders for data input
    with tf.variable_scope("data_input_placeholders") as data_input_placeholders_scope:
        # input in batch-major format: batch_size x g_seq_len
        gra_inputs = tf.placeholder(tf.int32, (None, None), name='grapheme_inputs')
        # variable length grapheme sequences with shape batch_size
        gra_input_lens = tf.placeholder(tf.int32, (None), name='grapheme_seq_lengths')

        # output of decoder will be the phonemes also with shape batch_size x p_seq_len
        dec_pho_inputs = tf.placeholder(tf.int32, (None, None), name='phoneme_decoder_inputs')
        # variable length phoneme sequences with shape batch_size
        dec_pho_inputs_lens = tf.placeholder(tf.int32, (None), name='phoneme_decoder_input_lengths')

        # labels (teacher forcing) with shape batch_size x p_seq_len
        pho_labels = tf.placeholder(tf.int32, (None, None), name='phoneme_labels')

    # Embedding layers
    with tf.variable_scope("embeddings") as embedding_scope:
        gra_embeddings = tf.Variable(tf.random_uniform([gra_vocab_size, embedding_dim], -1.0, 1.0), dtype=tf.float32,
                                     name='grapheme_embedding')
        # gra_inputs_embedded: [batch_size, time_step, embedding_dim] -> batch major format
        gra_inputs_embedded = tf.nn.embedding_lookup(gra_embeddings, gra_inputs)

        pho_embeddings = tf.Variable(tf.random_uniform([pho_vocab_size, embedding_dim], -1.0, 1.0), dtype=tf.float32,
                                     name='phoneme_embedding')
        # pho_output_embedded: [batch_size, time_step, embedding_dim] -> batch major format
        dec_pho_inputs_embedded = tf.nn.embedding_lookup(pho_embeddings, dec_pho_inputs)

    # create encoder and decoder LSTMs
    with tf.variable_scope("encoding") as encoding_scope:
        lstm_enc = tfc.rnn.BasicLSTMCell(num_hidden_units)

        # Dropout (= 1 - keep_prob)
        if use_dropout:
            dropout = 1 - keep_prob
            if dropout < 0.0:
                dropout = .2
                keep_prob = 1.0 - dropout
            lstm_enc = tf.contrib.rnn.DropoutWrapper(cell=lstm_enc, input_keep_prob=keep_prob)

        _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=gra_inputs_embedded, sequence_length=gra_input_lens,
                                          dtype=tf.float32)

    with tf.variable_scope("decoding") as decoding_scope:
        # encoder initial state is last_state of encoder
        lstm_dec = tfc.rnn.BasicLSTMCell(num_hidden_units)

        # Dropout (= 1 - keep_prob)
        if use_dropout:
            dropout = 1 - keep_prob
            if dropout < 0.0:
                dropout = .2
                keep_prob = 1.0 - dropout
            lstm_dec = tf.contrib.rnn.DropoutWrapper(cell=lstm_enc, input_keep_prob=keep_prob)

        dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=dec_pho_inputs_embedded,
                                           sequence_length=dec_pho_inputs_lens,
                                           initial_state=last_state, dtype=tf.float32)

    # output projection
    with tf.name_scope("output_projection"):
        logits = tfc.layers.fully_connected(dec_outputs, num_outputs=pho_vocab_size, activation_fn=tf.nn.softmax)
    """
        weights = tf.Variable(tf.random_uniform([num_hidden_units, pho_vocab_size], -0.01, 0.01, dtype=tf.float32))
        b = tf.Variable(tf.random_uniform([pho_vocab_size], -0.01, 0.01, dtype=tf.float32))
        predictions = tf.add(tf.matmul(dec_outputs, weights), b)
    """

    logits_argmax = tf.argmax(logits, axis=-1)

    with tf.name_scope("optimization"):
        # Loss function
        # TODO

        # get dynamic batch_size
        batch_size = tf.shape(gra_inputs)[0]
        # get dynamic output seq len
        pho_output_len = tf.shape(dec_pho_inputs)[0]

        loss = tfc.seq2seq.sequence_loss(logits, pho_labels, tf.ones([batch_size, pho_output_len]),
                                         average_across_batch=True, average_across_timesteps=True)
        tf.summary.scalar('loss', loss)

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    return gra_inputs, gra_input_lens, dec_pho_inputs, dec_pho_inputs_lens, pho_labels, optimizer, loss, logits, logits_argmax


def generate_batch_data(gra_seqs, pho_seqs, batch_size=1):
    start = 0
    shuffle = np.random.permutation(len(pho_seqs))
    gra_seqs = gra_seqs[shuffle]
    pho_seqs = pho_seqs[shuffle]

    while start + batch_size <= len(gra_seqs):
        enc_inputs = []
        enc_input_lens = []
        dec_inputs = []
        dec_input_lens = []
        labels = []
        for g in gra_seqs[start:start + batch_size]:
            enc_inputs.append(g)
            enc_input_lens.append(len(g))

        for p in pho_seqs[start:start + batch_size]:
            # dec_inputs doesn't contain last char -> end_token
            dec_inputs.append(p[:-1])
            # since pho_seqs have an appended <GO> token, the length has to be decreased by 1
            dec_input_lens.append(len(p) - 1)
            # labels doesn't contain first char -> go_token
            labels.append(p[1:])

        enc_inputs = np.array(enc_inputs)
        enc_input_lens = np.array(enc_input_lens)

        dec_inputs = np.array(dec_inputs)
        dec_input_lens = np.array(dec_input_lens)

        labels = np.array(labels)

        yield enc_inputs, enc_input_lens, dec_inputs, dec_input_lens, labels
        start += batch_size


def update_test_dec_input(old_dec_input, new_dec_prediction, batch_size=1):
    # init dec input (with <GO> token!)
    if old_dec_input is None:
        old_dec_input = np.zeros((batch_size, 1)) + new_dec_prediction
        return old_dec_input

    # new decoder input is the old decoder input PLUS the newly predicted output of the decoder
    new_dec_input = np.hstack([old_dec_input, new_dec_prediction[:, None]])

    return new_dec_input


def generate_model_path(learning_rate, num_hidden_units, embedding_dim, batch_size, keep_prob, base="/home/p0w3r/"):
    return base + "seq2seq_g2p_lr{}_hidden{}_ed{}_bs{}_kp{}/".format(learning_rate, num_hidden_units, embedding_dim,
                                                                     batch_size,
                                                                     keep_prob)


def log(msg):
    print(msg)
    logging.info(msg)


def main(data_path, ldr):
    # prepare the data that it fits into the LSTMs inputs
    grapheme_vocab_size, phoneme_vocab_size, grapheme_vocab_encoder, phoneme_vocab_encoder, encoded_graphemes, encoded_phonemes = prepare_data(
        data_path,
        add_go_token=True,
        add_end_token=True,
        apply_padding=False,
        use_numpy_arrays=True
    )

    # split test and train data
    grapheme_sequences_train, grapheme_sequences_test, phoneme_sequences_train, phoneme_sequences_test = train_test_split(
        encoded_graphemes, encoded_phonemes)

    # hyper parameters
    learning_rate = 0.001
    num_hidden_units = 100
    embedding_dim = 10
    epochs = 20
    batch_size = 1
    use_dropout = True
    keep_prob = .8

    log_data_root = generate_model_path(learning_rate, num_hidden_units, embedding_dim, batch_size, keep_prob, base_path=ldr)
    if not os.path.exists(log_data_root):
        os.makedirs(log_data_root)

    model_saver_path = log_data_root + 'model.ckpt'
    log_file_path = log_data_root + 'log.txt'

    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)

    gra_inputs, gra_input_lens, dec_pho_inputs, dec_pho_input_lens, pho_labels, optimizer, loss, logits, logits_argmax = build_graph(
        grapheme_vocab_size,
        phoneme_vocab_size,
        embedding_dim=embedding_dim,
        num_hidden_units=num_hidden_units,
        learning_rate=learning_rate,
        use_dropout=use_dropout,
        keep_prob=keep_prob,
    )

    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    summary_merge_op = tf.summary.merge_all()

    # start training
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter(model_saver_path, graph=sess.graph)

        for epoch_i in range(epochs):
            accuracies = []
            batch_losses = []
            log("Epoch {:3}: ".format(epoch_i))
            start = time.time()
            for batch_i, (
                    input_batch, input_lens_batch, dec_input_batch, dec_input_lens_batch, label_batch) in enumerate(
                generate_batch_data(grapheme_sequences_train, phoneme_sequences_train, batch_size)):
                # build feed dict
                f_dict = {gra_inputs: input_batch,
                          gra_input_lens: input_lens_batch,
                          dec_pho_inputs: dec_input_batch,
                          dec_pho_input_lens: dec_input_lens_batch,
                          pho_labels: label_batch}

                _, batch_loss, batch_logits, batch_logits_argmax = sess.run([optimizer, loss, logits, logits_argmax],
                                                                            feed_dict=f_dict)
                batch_accuracy = np.mean(batch_logits.argmax(axis=-1) == label_batch)
                accuracies.append(batch_accuracy)
                batch_losses.append(batch_loss)
            # merged_summary = sess.run(summary_merge_op)
            # writer.add_summary(summary=merged_summary, global_step=epoch_i)
            log(
                'Epoch {:3} Mean Epoch Loss: {:>6.3f} Mean Epoch Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(
                    epoch_i,
                    np.mean(
                        batch_losses),
                    np.mean(
                        accuracies),
                    time.time() - start)
            )
            save_path = saver.save(sess, model_saver_path, global_step=epoch_i)
            log('Saving model parameters in file: ' + model_saver_path)
            log("learning rate: {}".format(learning_rate))
            batch_losses.clear()
            accuracies.clear()

        # start testing
        log('Done with training! Starting decoding on test set!')
        test_accuracies = []
        for train_batch_i, (
                input_batch, input_lens_batch, dec_input_batch, dec_input_lens_batch, label_batch) in enumerate(
            generate_batch_data(grapheme_sequences_test, phoneme_sequences_test, batch_size)):

            # init dec input with <GO> token
            dec_input = update_test_dec_input(None, phoneme_vocab_encoder.encode('<GO>'), batch_size=batch_size)

            # predict new phonemes until <END> token was output before
            # TODO ONLY WORKS WITH BATCH SIZE 1! how can i stop predicting independently for each element in batch?
            while phoneme_vocab_encoder.encode('<END>') not in dec_input[:][-1]:
                # build training feed dict without labels
                f_dict_test = {gra_inputs: dec_input,
                               gra_input_lens: input_lens_batch,
                               dec_pho_inputs: dec_input_batch,
                               dec_pho_input_lens: dec_input_lens_batch}

                test_batch_logits, test_batch_logits_argmax = sess.run([logits, logits_argmax], feed_dict=f_dict_test)
                test_batch_accuracy = np.mean(test_batch_logits.argmax(axis=-1) == label_batch)
                test_accuracies.append(test_batch_accuracy)

                dec_input = update_test_dec_input(None, phoneme_vocab_encoder.encode('<GO>'))

            log('Mean Test Accuracy on test set is: {:>6.3f}'.format(np.mean(test_accuracies)))


if __name__ == '__main__':
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    main(data_path=sys.argv[1], log_data_root=sys.argv[2])
