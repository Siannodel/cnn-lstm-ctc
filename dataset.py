import generate_captch
import tensorflow as tf


class CaptchData():
    def __init__(self, config):
        self.config = config
        self.generator = generate_captch.CaptchaGenerator(
            char_set=self.config.char_set,
            lengths=self.config.text_lengths,
            shape=self.config.image_shape)
    def train_input_fn(self):
        dataset = tf.data.Dataset.from_generator(
            self.generator,output_types=(tf.float32,tf.int32,tf.int32))
        dataset = dataset.batch(self.config.batch_size)
        train_iterator = dataset.make_one_shot_iterator()
        images,labels,seq_lens = train_iterator.get_next()
        images = images*(2./255)-1
        batch = {
            'image': images,
            'label':labels,
            'seq_len': seq_lens,
        }
        return batch

    def test_input_fn(self):
        pass
        